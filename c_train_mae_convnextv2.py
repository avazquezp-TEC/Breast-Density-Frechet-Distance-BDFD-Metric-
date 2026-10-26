"""
Preentrenamiento auto-supervisado (SSL) de ConvNeXt V2 mediante enmascarado
denso estilo SimMIM, sobre las imágenes de MammoBench que c_dataset_manage.py
descarta (vista MLO, vista vacía, o sin etiqueta de densidad).

Solo backbone, sin decoder ni cabeza de clasificación se guarda en disco
 para ser cargado como inicialización de
c_train_convnext.py.
"""

from __future__ import annotations
from tqdm import tqdm
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from c_dataset_manage import MammoDataset

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimMIMMaskGenerator:
    """
    Genera una máscara binaria a nivel de "parche" vectorizada para todo el batch
    directamente en el dispositivo (GPU), evitando cuellos de botella.
    """
    def __init__(self, input_size: int = 512, mask_patch_size: int = 32, mask_ratio: float = 0.6):
        assert input_size % mask_patch_size == 0, "input_size debe ser múltiplo de mask_patch_size"
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.grid_size = input_size // mask_patch_size          
        self.num_patches = self.grid_size ** 2                  
        self.mask_ratio = mask_ratio
        self.num_mask = int(round(self.num_patches * mask_ratio))

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Genera ruido aleatorio directamente en GPU para todo el batch
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_shuffle[:, :self.num_mask], 1.0)
        
        mask = mask.view(batch_size, self.grid_size, self.grid_size)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=1)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=2)
        return mask  # (B, H, W) en GPU


class ConvNeXtV2SimMIM(nn.Module):
    """
    Preentrenamiento SimMIM sobre un encoder ConvNeXt V2 de
    timm.

    1. Sustituye los parches enmascarados de la imagen por un mask-token
       aprendible (broadcast por canal), ANTES del encoder.
    2. Pasa la imagen (parcialmente enmascarada) por el encoder completo.
    3. Un decoder ligero (1x1 conv + PixelShuffle, sin capas pesadas tipo
       transformer) reconstruye la imagen a partir del mapa de
       características final.
    """
    def __init__(self, encoder_name: str = "convnextv2_tiny.fcmae", pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)

        last_stage = self.encoder.feature_info[-1]
        self.encoder_stride = last_stage["reduction"]     # 32 para *_tiny / *_base
        self.encoder_channels = last_stage["num_chs"]     # 768 para *_tiny

        self.mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.encoder_channels, (self.encoder_stride ** 2) * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, 3, H, W) ya normalizado (ImageNet mean/std)
        # mask: (B, H, W), 1.0 = parche oculto
        mask_c = mask.unsqueeze(1)                                   # (B, 1, H, W)
        masked_x = x * (1 - mask_c) + self.mask_token * mask_c       # sustituye los parches ocultos

        features = self.encoder.forward_features(masked_x)           # (B, C, H/stride, W/stride)
        recon = self.decoder(features)                               # (B, 3, H, W)
        return recon, mask


def simmim_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    L1 sobre los píxeles reconstruidos, calculada SOLO sobre la región
    enmascarada. Si se calculara sobre toda la imagen, el modelo podría
    "hacer trampa" simplemente copiando los píxeles visibles.
    """
    mask_c = mask.unsqueeze(1)
    loss = F.l1_loss(recon, target, reduction="none")
    loss = (loss * mask_c).sum() / (mask_c.sum() * target.shape[1] + 1e-5)
    return loss


def train_one_epoch(model, loader, mask_gen, optimizer, scaler, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for images in tqdm(loader, desc="Entrenando", unit="batch"):
        images = images.to(device, non_blocking=True)
        b = images.size(0)
        mask = mask_gen(b, device)

        optimizer.zero_grad()
        
        #with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            recon, mask = model(images, mask)
            loss = simmim_loss(recon, images, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * b
        total += b

    return running_loss / total


@torch.no_grad()
def validate(model, loader, mask_gen, device) -> float:
    model.eval()
    running_loss = 0.0
    total = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        b = images.size(0)
        
        # NUEVA FORMA: Llamada vectorizada a mask_gen
        mask = mask_gen(b, device)
        
        # Mantenemos autocast en validación para no quedarnos sin memoria
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            recon, mask = model(images, mask)
            loss = simmim_loss(recon, images, mask)
            
        running_loss += loss.item() * b
        total += b
    return running_loss / total


def main(
    seed: int = 42,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    mask_ratio: float = 0.6,
    csv_path: str = "../../Experimentos/Mammo_Bench_v2/mammo-bench.csv",
    out_path: str = "convnextv2_tiny_mae_encoder.pth",
):
    seed_everything(seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {seed}")

    # dataset = MammoSSLDataset(csv_file=Path(csv_path))
    dataset = MammoDataset(csv_file = Path(csv_path), ssl_mode=True, crop=False)
    logger.info(f"Imágenes disponibles para SSL: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(
            "MammoSSLDataset quedó vacío. Revisa que el CSV tenga filas con "
            "vista MLO, vista vacía o sin etiqueta de densidad."
        )

    train_loader, val_loader = dataset.get_loader(seed=seed, batch_size=batch_size)

    # Se parte del checkpoint de ConvNeXt V2 preentrenado con FCMAE en
    # ImageNet 
    model = ConvNeXtV2SimMIM(encoder_name="convnextv2_tiny.fcmae", pretrained=True).to(device)
    mask_gen = SimMIMMaskGenerator(input_size=512, mask_patch_size=model.encoder_stride, mask_ratio=mask_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # 1. Inicializar el GradScaler para mixed precision
    #scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val = float("inf")
    best_state = None
    counter = 0
    for epoch in range(1, max_epochs + 1):
        # 2. Agregar scaler a la llamada
        train_loss = train_one_epoch(model, train_loader, mask_gen, optimizer, scaler, device)
        val_loss = validate(model, val_loader, mask_gen, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        improved = val_loss < (best_val - 1e-4)
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping en epoch {epoch}. Mejor Val Loss: {best_val:.4f}")
                break
    else:
        logger.info(f"Se alcanzó max_epochs={max_epochs} sin activar early stopping.")

    if best_state is not None:
        torch.save(best_state, out_path)
        logger.info(f"Encoder SSL guardado en: {out_path}")
        logger.info("Cárgalo en b_train_convnext.py pasando esta ruta como ssl_checkpoint.")


if __name__ == "__main__":
    logging.basicConfig(
        filename="ejecucionSSL.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Preentrenamiento SSL (SimMIM) de ConvNeXt V2 sobre MammoBench.")
    parser.add_argument("--csv", type=str, default="../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    parser.add_argument("--epochs", type=int, default=200) #200
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.7)
    parser.add_argument("--out", type=str, default="convnextv2_tiny_mae_encoder.pth")
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        out_path=args.out,
    )
