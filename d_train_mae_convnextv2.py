"""Self-supervised ConvNeXt V2 pretraining using dense SimMIM-style masking.

It uses DrVinMammo images for SSL

The training logic lives in :func:`fit`, which is reused by ``c_optuna_ssl.py``
for hyperparameter search. Run this file directly for a full training run,
optionally with ``--params-json best_params.json`` produced by the Optuna study.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from d_dataset_manage import IMAGE_SIZE, MammoDataset

logger = logging.getLogger(__name__)

SCHEDULERS = ("cosine", "linear", "onecycle", "plateau")


def seed_everything(seed: int = 42) -> None:
    """Seed all supported random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    # if torch.cuda.is_available():
    #     return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


class SimMIMMaskGenerator:
    """Generate vectorized patch-level binary masks directly on the device.

    If ``seed`` is given, the masks are reproducible: call :meth:`reset` before
    each pass over a fixed loader to get exactly the same masks every time.
    """

    def __init__(
        self,
        input_size: int = 512,
        mask_patch_size: int = 32,
        mask_ratio: float = 0.6,
        seed: int | None = None,
    ) -> None:
        assert input_size % mask_patch_size == 0, (
            "input_size must be a multiple of mask_patch_size"
        )
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.grid_size = input_size // mask_patch_size
        self.num_patches = self.grid_size**2 # defines how many patches can be created considering input size and patch size
        self.mask_ratio = mask_ratio
        self.num_mask = max(1, int(round(self.num_patches * mask_ratio)))
        self.seed = seed
        self._generator: torch.Generator | None = None

    def reset(self) -> None:
        """Restart the reproducible mask sequence (no-op when unseeded)."""
        self._generator = None

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a binary mask for a batch of images."""
        generator = None
        if self.seed is not None:
            if self._generator is None:
                self._generator = torch.Generator(device=device).manual_seed(self.seed)
            generator = self._generator

        noise = torch.rand(
            batch_size, self.num_patches, device=device, generator=generator
        )
        ids_shuffle = torch.argsort(noise, dim=1)

        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask.scatter_(1, ids_shuffle[:, : self.num_mask], 1.0)

        mask = mask.view(batch_size, self.grid_size, self.grid_size)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=1)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=2)
        return mask  # (B, H, W) on the device.


class ConvNeXtV2SimMIM(nn.Module):
    """Perform SimMIM pretraining with a ConvNeXt V2 encoder.

    1. Replace masked image patches with a learnable mask token, broadcast
       across channels, before the encoder.
    2. Pass the partially masked image through the complete encoder.
    3. Reconstruct the image from the final feature map with a lightweight
       decoder consisting of a 1x1 convolution and PixelShuffle.
    """

    def __init__(
        self,
        encoder_name: str = "convnextv2_base.fcmae",
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            num_classes=0, # deletes the final clasification layer
            in_chans=1,     # adjust the first layer to accept only gray scale images
            drop_path_rate=drop_path_rate, # to avoid overfitting uses turn off neurons
        )
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing(True)

        last_stage = self.encoder.feature_info[-1] # ask for the features of last layer
        self.encoder_stride = last_stage["reduction"]
        self.encoder_channels = last_stage["num_chs"]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1)) # helps to reconstruc the missing square of the MAE
        nn.init.trunc_normal_(self.mask_token, std=0.02) # this reconstruccion is performed in forward

        self.decoder = nn.Sequential( # changes the tensor (B,last_stage["num_chs"],H/last_stage["reduction"],W/last_stage["reduction"])
            nn.Conv2d(                  # to a image with the convolution and pixel suffle
                self.encoder_channels,
                self.encoder_stride**2,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct masked images and return the mask used."""
        # x: (B, 1, H, W), normalized to the [0, 1] range.
        # mask: (B, H, W), where 1.0 indicates a hidden patch.
        mask_c = mask.unsqueeze(1)  # (B, 1, H, W)
        masked_x = x * (1 - mask_c) + self.mask_token * mask_c

        features = self.encoder.forward_features(masked_x)
        recon = self.decoder(features)
        return recon, mask


def simmim_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute L1 loss only over reconstructed pixels in masked regions."""
    # Computing loss over the full image would let the model copy visible pixels.
    mask_c = mask.unsqueeze(1)
    loss = F.l1_loss(recon.float(), target, reduction="none")
    loss = (loss * mask_c).sum() / (mask_c.sum() * target.shape[1] + 1e-5)
    return loss


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> tuple[object, str]:
    """Create an LR scheduler sized for the *actual* number of epochs.

    Returns ``(scheduler, step_mode)`` where step_mode is ``"epoch"``,
    ``"batch"`` or ``"plateau"`` (the last one needs the validation loss).
    """
    if name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Choose from {SCHEDULERS}.")

    sched = torch.optim.lr_scheduler
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))

    if name == "onecycle":
        max_lr = optimizer.param_groups[0]["lr"]
        pct_start = min(0.5, max(0.05, warmup_epochs / epochs))
        scheduler = sched.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=epochs * steps_per_epoch,
            pct_start=pct_start,
            cycle_momentum=False,
        )
        return scheduler, "batch"

    if name == "plateau":
        scheduler = sched.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )
        return scheduler, "plateau"

    decay_epochs = max(1, epochs - warmup_epochs)
    if name == "cosine":
        main = sched.CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    else:  # linear
        main = sched.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=decay_epochs
        )

    if warmup_epochs > 0:
        warmup = sched.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        main = sched.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_epochs]
        )
    return main, "epoch"


def train_one_epoch(
    model,
    loader,
    mask_gen,
    optimizer,
    scaler,
    device,
    scheduler=None,
    step_per_batch: bool = False,
    grad_clip: float = 0.0,
    show_progress: bool = True,
) -> float:
    """Train the model for one epoch and return the mean loss."""
    model.train()
    amp_enabled = device.type == "cuda"
    running_loss = 0.0
    total = 0

    for images in tqdm(loader, desc="Training", unit="batch", disable=not show_progress, leave=False):
        images = images.to(device, non_blocking=True)
        b = images.size(0)
        mask = mask_gen(b, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled):
            recon, mask = model(images, mask)
            loss = simmim_loss(recon, images, mask)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and step_per_batch:
            scheduler.step()

        running_loss += loss.item() * b
        total += b

    return running_loss / max(total, 1)


@torch.no_grad()
def validate(model, loader, mask_gen, device) -> float:
    """Evaluate the model and return the mean validation loss.

    ``mask_gen`` should be seeded: it is reset here so every call sees the same
    masks, which makes the validation loss comparable across epochs and trials.
    """
    model.eval()
    amp_enabled = device.type == "cuda"
    mask_gen.reset()
    running_loss = 0.0
    total = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        b = images.size(0)
        mask = mask_gen(b, device)

        # Keep autocast during validation to avoid running out of memory.
        with torch.autocast(
            device_type="cuda" if amp_enabled else "cpu", enabled=amp_enabled
        ):
            recon, mask = model(images, mask)
            loss = simmim_loss(recon, images, mask)

        running_loss += loss.item() * b
        total += b
    return running_loss / max(total, 1)


def fit(
    model: ConvNeXtV2SimMIM,
    train_loader,
    val_loader,
    train_mask_gen: SimMIMMaskGenerator,
    val_mask_gen: SimMIMMaskGenerator,
    device: torch.device,
    *,
    max_epochs: int,
    lr: float,
    weight_decay: float = 0.05,
    scheduler_name: str = "cosine",
    warmup_epochs: int = 0,
    grad_clip: float = 0.0,
    patience: int | None = None,
    keep_state: bool = True,
    on_epoch_end: Callable[[int, float], None] | None = None,
    show_progress: bool = True,
) -> tuple[float, int, dict | None]:
    """Train with the given hyperparameters.

    ``on_epoch_end(epoch, val_loss)`` is called after every validation and may
    raise (e.g. ``optuna.TrialPruned``). ``patience=None`` disables early
    stopping. Returns ``(best_val_loss, best_epoch, best_encoder_state)``.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler, step_mode = build_scheduler(
        scheduler_name,
        optimizer,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=warmup_epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    counter = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            train_mask_gen,
            optimizer,
            scaler,
            device,
            scheduler=scheduler,
            step_per_batch=(step_mode == "batch"),
            grad_clip=grad_clip,
            show_progress=show_progress,
        )
        val_loss = validate(model, val_loader, val_mask_gen, device)

        if step_mode == "epoch":
            scheduler.step()
        elif step_mode == "plateau":
            scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_loss)

        if not math.isfinite(val_loss):
            logger.warning("Non-finite validation loss; stopping.")
            break

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            counter = 0
            if keep_state:
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.encoder.state_dict().items()
                }
        else:
            counter += 1
            if patience is not None and counter >= patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best validation loss: {best_val:.4f}"
                )
                break

    return best_val, best_epoch, best_state


def run(args: argparse.Namespace) -> None:
    """Run a full SSL pretraining with the hyperparameters in ``args``."""
    seed_everything(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Hyperparameters: {vars(args)}")

    dataset = MammoDataset(csv_file=Path(args.csv), ssl_mode=True, crop=False)
    logger.info(f"Images available for SSL: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(
            "The SSL dataset is empty. Check the CSV path and that the DICOM "
            "files exist under <csv_dir>/images/<study_id>/<image_id>.dicom."
        )

    train_loader, val_loader = dataset.get_loader(seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers )

    model = ConvNeXtV2SimMIM(
        encoder_name=args.encoder,
        pretrained=True,
        drop_path_rate=args.drop_path,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)

    mask_patch_size = args.mask_patch_size or model.encoder_stride
    if mask_patch_size % model.encoder_stride != 0:
        raise ValueError(
            f"mask_patch_size ({mask_patch_size}) must be a multiple of the "
            f"encoder stride ({model.encoder_stride})."
        )
    train_mask_gen = SimMIMMaskGenerator(
        IMAGE_SIZE, mask_patch_size, args.mask_ratio
    )
    # Fixed validation protocol, independent of the training mask settings.
    val_mask_gen = SimMIMMaskGenerator(# here the mask_patch_size is the minimum: model_encoder_stride
        IMAGE_SIZE, model.encoder_stride, args.val_mask_ratio, seed=args.seed
    )

    _, best_epoch, best_state = fit(
        model,
        train_loader,
        val_loader,
        train_mask_gen,
        val_mask_gen,
        device,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        patience=args.patience,
    )

    if best_state is not None:
        torch.save(best_state, args.out)
        logger.info(f"Saved SSL encoder (epoch {best_epoch}) to: {args.out}")
        logger.info(
            "Load it in c_train_convnext.py by passing this path as ssl_checkpoint."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SSL (SimMIM) pretraining of ConvNeXt V2 on MammoBench."
    )
    parser.add_argument("--csv", type=str, default="../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    parser.add_argument("--encoder", type=str, default="convnextv2_base.fcmae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--mask-patch-size", type=int, default=None, help="Multiple of the encoder stride (32). Default: the stride." ) # size of the mask
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.0, help = "turn off neuron branch to prevent overfitting") 
    parser.add_argument("--val-mask-ratio", type=float, default=0.6)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--params-json", type=str, default=None, help="JSON with hyperparameters (e.g. best_params.json from Optuna) that override the command-line values." )
    parser.add_argument("--out", type=str, default="convnextv2_base_mae_encoder.pth")
    parser.add_argument("--log-file", type=str, default="ejecucionSSL_standar.log")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.params_json:
        with open(args.params_json) as fh:
            for key, value in json.load(fh).items():
                if not hasattr(args, key):
                    raise KeyError(f"Unknown hyperparameter in JSON: {key}")
                setattr(args, key, value)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    run(args)
