import copy
import random
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models import ConvNeXt_Base_Weights, convnext_base
from torch.utils.data import DataLoader, random_split
import logging
import pandas as pd
from b_dataset_manage import MammoDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix




def seed_everything(seed: int = 42) -> None:
    """
    Fija todas las semillas relevantes para que el entrenamiento sea
    reproducible entre corridas (split train/val, sampler, init de pesos
    de la nueva capa lineal, orden de batches, etc.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Fuerza operaciones deterministas en cudnn (a costa de algo de velocidad)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_device() -> torch.device:           # define el mejor dispositivo disponible
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_model(num_classes: int) -> tuple[nn.Module, object]:
    """
    La parte clasificatoria de convnext es.
            (classifier): Sequential(
            (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True, bias=True)
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): Linear(in_features=768, out_features=1000, bias=True)
            )
        )
    por eso usa indice 2, para modificar solo la ultima capa
    """
    weights = ConvNeXt_Tiny_Weights.DEFAULT                         # descarga los pesos de la convNext tiny
    model = convnext_tiny(weights=weights)                          # carga esos pesos el modelo tiny
    # weights = ConvNeXt_Base_Weights.DEFAULT
    # model = convnext_base(weights=weights)
    logger.debug(model)                                             # muestra la estructura del modelo
    in_features = model.classifier[2].in_features                   # obtiene el tamaño de la capa final
    # Reemplazar la capa final por Dropout + Linear
    model.classifier[2] = nn.Sequential(                            # reemplaza classifier[2] (head) por la nueva capa
        nn.Dropout(p=0.3),                                          # probabilidad de apagar neuronas
        nn.Linear(in_features, num_classes)                         #  y con entrada (in_features) y salida(num_clases)
    )

    return model, weights


def get_optimizer(model: nn.Module, backbone_lr: float, head_lr: float):
    """
    Esta función asigna 2 Learning rates al modelo. 
    Uno pequeño para el backbone y otro mas grande para la capa final.
    Esto podira reemplazarlo con AutoML
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "classifier.2" in name:
            head_params.append(param)                       # si es clasiffier lo pone en la lista head_params
        else:
            backbone_params.append(param)                   # si no en backbone

    return torch.optim.AdamW(                               # asigna los lr a cada lista
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    )

def preprocess_image(image_path: Path, weights, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    x = weights.transforms()(image).unsqueeze(0).to(device)
    return x

@torch.inference_mode()
def infer(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(image_tensor)

class EarlyStopping:
    """
    Detiene el entrenamiento cuando la pérdida de validación deja de mejorar
    durante `patience` épocas seguidas. También guarda en memoria los pesos
    del mejor modelo visto hasta el momento, para restaurarlos al final.
    Esto es lo que permite no fijar a mano el número de épocas: se entrena
    "hasta convergencia" dentro de un límite máximo de seguridad (max_epochs).
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience  # True -> hay que detenerse

def compute_metrics(y_true, y_pred, num_classes: int = 4) -> dict:
    """
    Calcula métricas estándar de clasificación multiclase sobre el set de
    validación. macro-average trata a las 4 categorías (A,B,C,D) por igual,
    lo cual es preferible aquí porque el dataset está desbalanceado.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": acc,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": cm,
    }

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes: int = 4):
    model.eval()  # Cambia a modo evaluación (desactiva Dropout, Batch Normalization, etc.)

    running_loss = 0.0
    total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:   # torch.no_grad() evita calcular gradientes y optimiza memoria/tiempo
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        # Acumula la pérdida ponderada por el tamaño del lote
        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total
    metrics = compute_metrics(all_labels, all_preds, num_classes=num_classes)
    return avg_loss, metrics


def main(seed: int = 42, max_epochs: int = 50, patience: int = 7, batch_size: int = 32):

    seed_everything(seed)                                                   # reproducibilidad total

    device = get_device()                                                   # select device for tensor.
    model, weights = build_model(num_classes=4)                             # Remove head layer and replace for a lineal layer for new classification
    model = model.to(device)                                                # transfers to best device
    logger.info(f"Using device: {device}")
    logger.info(f"Seed: {seed}")

    optimizer = get_optimizer(model, backbone_lr=1e-5, head_lr=1e-3)        # asign differents lr to the model
    criterion = nn.CrossEntropyLoss()                                       # asigna criterio para loss function

    # Ajusta automáticamente el learning rate (el "step size") cuando la
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(                 # pérdida de validación se estanca: reduce ambos grupos de lr a la mitad
        optimizer, mode="min", factor=0.5, patience=3                       # tras 3 épocas sin mejora.
    )

    ruta_csv = Path("../../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    dataset = MammoDataset(csv_file = ruta_csv)                             # crea el objeto MammoDataset que creara los loaders de entrenamiento y validacion

    train_loader, val_loader = dataset.get_loader(seed=seed, batch_size=batch_size) # invoca al mammodataset para crear los loaders de entrenamiento y validación

    early_stopping = EarlyStopping(patience=patience)                       # configura early stoping con patience de 7. pero a que factor?

    for epoch in range(1, max_epochs + 1):                                  # hace entrenamiento y validacion en cada epoca
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)                                            # decide si baja el lr
        head_lr = optimizer.param_groups[1]["lr"]                           # de head
        backbone_lr = optimizer.param_groups[0]["lr"]                       # o backbone

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val F1(macro): {val_metrics['f1']:.2f}% | "
            f"lr(head/backbone): {head_lr:.2e}/{backbone_lr:.2e}"
        )

        stop = early_stopping.step(val_loss, model)
        if stop:
            logger.info(
                f"Early stopping en epoch {epoch} (sin mejora en {patience} "
                f"épocas). Mejor Val Loss: {early_stopping.best_loss:.4f}"
            )
            break
    else:
        logger.info(f"Se alcanzó max_epochs={max_epochs} sin activar early stopping.")

    # Restaura los pesos del mejor checkpoint visto durante el entrenamiento
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    torch.save(model.state_dict(), "convnext_tiny_custom.pth")

    # Validación final con el mejor modelo
    final_loss, final_metrics = validate(model, val_loader, criterion, device)

    logger.info(
        f"Validation Stage (mejor modelo) -> Loss: {final_loss:.4f} | "
        f"Accuracy: {final_metrics['accuracy']:.2f}% | "
        f"Precision(macro): {final_metrics['precision']:.2f}% | "
        f"Recall(macro): {final_metrics['recall']:.2f}% | "
        f"F1(macro): {final_metrics['f1']:.2f}%"
    )
    logger.info(
        "Matriz de confusión (filas=real, columnas=predicho, orden A,B,C,D):\n"
        f"{final_metrics['confusion_matrix']}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)        # define el logger para mostrar información en pantalla
    main(max_epochs = 1)