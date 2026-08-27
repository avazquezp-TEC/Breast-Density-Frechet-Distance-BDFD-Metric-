import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import logging

class ResizeWithPadding:
    """
    Redimensiona la imagen manteniendo su relación de aspecto original y
    rellena (letterbox) el espacio sobrante con `fill` hasta llegar a un
    cuadrado de `size` x `size` sin deformar la imagen.
    """
    def __init__(self, size=224, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        # padding = (izquierda, arriba, derecha, abajo)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, padding, fill=self.fill)

# Transformación básica para entrenamiento y validación
data_transforms = transforms.Compose([
    ResizeWithPadding(224, fill=0),  # Resize proporcional + padding negro (evita deformar la imagen)
    transforms.ToTensor(),          # Convierte PIL/NumPy a Tensor Y escala valores a [0.0, 1.0]
    transforms.Normalize(           # Normalización estándar de ImageNet (opcional pero muy recomendada)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Transformación para la etapa de preentrenamiento auto-supervisado (SSL).
# A diferencia de `data_transforms`, aquí sí conviene aumentar los datos
# (recorte aleatorio + flip) porque el objetivo no es clasificar sino que
# el modelo aprenda representaciones generales del tejido mamario.
ssl_transforms = transforms.Compose([
    ResizeWithPadding(224, fill = 0),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])


class MammoDataset(Dataset):
    class_to_idx = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    }
    def __init__(self, csv_file, transform=data_transforms):
        self.transform = transform
        self.root_dir = Path(csv_file).parent
        self.data = pd.read_csv(csv_file)
        
        # Mapear etiquetas
        self.data["label"] = self.data["density"].map(self.class_to_idx)                                # agrega una columna "label" cambiando letra por número
        self.data = self.data[["preprocessed_image_path", "mask_path", "laterality", "view", "label"]]  # resduce datos inncenesarios de la tabla
        
        # Filtrar datos: Mantener solo las filas donde view NO sea "MLO" o este vacia.
        self.data = self.data[self.data['view'] != 'MLO']
        self.data = self.data[self.data['view'].notna()]
        self.data = self.data[self.data['label'].notna()]
        
        self.data = self.data.reset_index(drop=True)                                                    # Reiniciar los índices del DataFrame, ya que si no quedan huecos
        self.logger = logging.getLogger(__name__)

    def __len__(self): # devuelve la cantidad de datos
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]                                               # iloc accede a indices del csv

        # devuelve un objeto del data set. iloc es un index para pandas
        img_path = os.path.join(self.root_dir, row['preprocessed_image_path'])  # combina las rutas
        image = Image.open(img_path).convert("RGB")                             # abre la imagen.

        mask_path = os.path.join(self.root_dir, row['mask_path'])
        mask = Image.open(mask_path).convert("L")                               # escala de grises

        image_np = np.array(image)
        mask_np = np.array(mask)
        if mask.size == image.size:
            coords = np.argwhere(mask_np > 0)                                   # Encontrar coordenadas donde la máscara es >0
            if len(coords) > 0:                                                 # Calcular bounding box
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0) + 1
                cropped = image_np[y0:y1, x0:x1]                                # Recortar la imagen
            else:
                cropped = image_np
        else:
            self.logger.debug("mascara invalida. saltando...")
            cropped = image_np

        cropped_img = Image.fromarray(cropped)                                  # 1. Convertir siempre a PIL para aplicarle las transformaciones de torchvision
        label = int(row['label'])

        
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)                           # 2. Aplicar transforms Para normalizar 
        else:
            cropped_img = transforms.ToTensor()(cropped_img)                    # Si no hay transform, al menos convertir a Tensor para evitar el crash del DataLoader

        return cropped_img, label

    
    def get_loader(self, split_rel=0.8, seed=42, batch_size=32, num_workers=0):
        # ⚠️ Reproducibilidad: generator con seed fija para el split y el sampler
        generator = torch.Generator().manual_seed(seed)
        train_size = int(split_rel * len(self))                                     # obtiene longitud del dataset al 80%
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(                                  # separa los datos con la seed
            self, [train_size, val_size], generator = generator
        )
        train_indices = train_dataset.indices                                       # Obtener etiquetas solo del Subconjunto de Entrenamiento
        train_labels = np.array(self.data.iloc[train_indices]["label"].values, dtype=np.uint8) # y de validación

        class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long)) # cuenta cuantos elementos hay por categoria, pero solo de train
        class_weights = 1. / class_counts.float()                                   # Clases con menos ejemplos reciben un peso mayor.
        # Las clases minoritarias tienen más probabilidad de ser seleccionadas, compensando el desbalance.
        sample_weights = [class_weights[label] for label in train_labels]           # El DataLoader usará este vector de pesos para elegir muestras.
        sampler = WeightedRandomSampler(                                            # crea el objeto sampler usando el vector de pesos para elegir las muestras de a
            weights=sample_weights,                                                 # cuerdo con la probabilidad antes calculada
            num_samples=len(sample_weights),
            replacement=True,                                                       # replacement=True significa que puede volver a elegir la misma muestra varias veces dentro de una época.
            generator=generator,
        )
        
        # dataset debe ser una clase que herede de Dataset y lea tus rutas desde el CSV. Usa random_split o train_test_split para separar entrenamiento/validación
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        return train_loader, val_loader


class MammoSSLDataset(Dataset):
    """
    Dataset para la etapa de preentrenamiento auto-supervisado (SSL) con
    ConvNeXt V2 + SimMIM.

    Usa exactamente el COMPLEMENTO del filtro de MammoDataset: vista MLO,
    vista vacía, o sin etiqueta de densidad (por Ley de De Morgan, el
    complemento de "view != MLO AND view.notna() AND label.notna()" es
    "view == MLO OR view.isna() OR label.isna()"). No requiere etiquetas,
    así que __getitem__ devuelve solo la imagen.

    A diferencia de MammoDataset, estas imágenes están menos curadas, así
    que el manejo de máscara/archivo faltante es más defensivo: si la
    imagen base no existe se descarta la fila en __init__, y si la
    máscara falta o es inválida simplemente se usa la imagen completa
    (sin recorte) en vez de fallar.
    """
    def __init__(self, csv_file, transform=ssl_transforms):
        self.transform = transform
        self.root_dir = Path(csv_file).parent
        self.logger = logging.getLogger(__name__)

        data = pd.read_csv(csv_file)
        data["label"] = data["density"].map(MammoDataset.class_to_idx)

        discarded_mask = (# selecciona las imagenes complemento
            (data["view"] == "MLO") | data["view"].isna() | data["label"].isna()
        )
        self.data = data.loc[discarded_mask, ["preprocessed_image_path", "mask_path"]].reset_index(drop=True)

        # Descarta filas cuya imagen base no existe en disco, para no
        # tronar el DataLoader en medio de un epoch por un path roto.
        exists = self.data["preprocessed_image_path"].apply(
            lambda p: isinstance(p, str) and os.path.isfile(os.path.join(self.root_dir, p))
        )
        n_before = len(self.data)
        self.data = self.data[exists].reset_index(drop=True)
        n_removed = n_before - len(self.data)
        if n_removed > 0:
            self.logger.warning(
                f"{n_removed} imágenes listadas en el CSV no existen en disco; se omiten del set SSL."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["preprocessed_image_path"])
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        cropped = image_np
        mask_path = row.get("mask_path")
        if isinstance(mask_path, str):
            full_mask_path = os.path.join(self.root_dir, mask_path)
            try:
                mask = Image.open(full_mask_path).convert("L")
                mask_np = np.array(mask)
                if mask.size == image.size:
                    coords = np.argwhere(mask_np > 0)
                    if len(coords) > 0:
                        y0, x0 = coords.min(axis=0)
                        y1, x1 = coords.max(axis=0) + 1
                        cropped = image_np[y0:y1, x0:x1]
                else:
                    self.logger.debug("mascara invalida (tamaño no coincide). usando imagen completa.")
            except (FileNotFoundError, OSError):
                self.logger.debug(f"mascara no encontrada para {img_path}. usando imagen completa.")

        cropped_img = Image.fromarray(cropped)
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)
        else:
            cropped_img = transforms.ToTensor()(cropped_img)
        return cropped_img

    def get_ssl_loader(self, val_rel=0.05, seed=42, batch_size=64, num_workers=0):
        """
        Split simple train/val (95/5 por defecto) solo para poder
        monitorear que la pérdida de reconstrucción en SimMIM no se
        estanque o colapse. No hace falta un sampler balanceado aquí
        porque no hay clases: es aprendizaje auto-supervisado.
        """
        generator = torch.Generator().manual_seed(seed)
        val_size = max(1, int(val_rel * len(self)))
        train_size = len(self) - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader


if __name__ == "__main__":
    
    ruta_csv = Path("../../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    dataset = MammoDataset(csv_file=ruta_csv, transform=data_transforms)
    train_loader, val_loader = dataset.get_loader()
    #aqui va train y val.

    ssl_dataset = MammoSSLDataset(csv_file=ruta_csv)
    logging.getLogger(__name__).info(
        f"Imágenes descartadas disponibles para SSL: {len(ssl_dataset)}"
    )