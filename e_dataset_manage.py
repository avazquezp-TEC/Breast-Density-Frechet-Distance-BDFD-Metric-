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
    def __init__(self, size=512, fill=0):
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
    ResizeWithPadding(512, fill=0),  # Resize proporcional + padding negro (evita deformar la imagen)
    transforms.ToTensor(),          # Convierte PIL/NumPy a Tensor Y escala valores a [0.0, 1.0]
    transforms.Normalize(           # Normalización estándar de ImageNet (opcional pero muy recomendada)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


ssl_transforms = transforms.Compose([
    # 1. Rotamos la imagen original (el tejido sin padding negro todavía).
    transforms.RandomRotation(degrees=(0, 90)),
    
    # 2. Escala la imagen manteniendo la relación de aspecto original.
    # El lado más corto será de 512px. El largo será > 512px.
    transforms.Resize(512), 
    
    # 3. Recorta un cuadrado exacto de 512x512 al azar.
    # Como el lado corto ya es 512, tomará el grosor completo y 
    # deslizará el recorte a lo largo del tejido restante sin tomar fondo negro.
    transforms.RandomCrop(512),

    # 4. Volteos direccionales
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    
    # 5. Formato estándar
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class MammoDataset(Dataset):
    class_to_idx = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    }
    def __init__(self, csv_file, transform=data_transforms, ssl_mode=False, crop = False):
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(csv_file).parent
        self.data = pd.read_csv(csv_file)
        self.ssl_mode = ssl_mode
        self.crop = crop

        # Mapear etiquetas
        self.data["label"] = self.data["density"].map(self.class_to_idx)                                # agrega una columna "label" cambiando letra por número. solo necesario para clasificación
        self.data = self.data[["preprocessed_image_path", "mask_path", "laterality", "view", "label"]]  # deja unicamente las columnas necesarias
        if not self.ssl_mode: # flujo para finetunning de clasificaciones
            self.transform = transform
            
            # Filtrar datos: Mantener solo las filas donde view NO sea "MLO" o este vacia.
            self.data = self.data[self.data['view'] != 'MLO']
            self.data = self.data[self.data['view'].notna()]
            self.data = self.data[self.data['label'].notna()]
        else: #flujo para SSL
            self.transform = ssl_transforms
            discarded_mask = (# selecciona las imagenes complemento
                        (self.data["view"] == "MLO") | self.data["view"].isna() | self.data["label"].isna()
                    )
            self.data = self.data.loc[discarded_mask, ["preprocessed_image_path", "mask_path"]]

        self.data = self.data.reset_index(drop=True)                                                    # Reiniciar los índices del DataFrame, ya que si no quedan huecos
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
        return
        

    def __len__(self): # devuelve la cantidad de datos
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]                                               # iloc accede a indices del csv

        # devuelve un objeto del data set. iloc es un index para pandas
        img_path = os.path.join(self.root_dir, row['preprocessed_image_path'])  # combina las rutas
        image = Image.open(img_path).convert("RGB")                             # abre la imagen.
        image_np = np.array(image)
        
        mask_path = os.path.join(self.root_dir, row['mask_path'])
        mask = Image.open(mask_path).convert("L")                               # escala de grises
        mask_np = np.array(mask)
        cropped = image_np
        if self.crop:
            try:
                mask = Image.open(mask_path).convert("L")
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

        cropped_img = Image.fromarray(cropped)                                  # 1. Convertir siempre a PIL para aplicarle las transformaciones de torchvision
        
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)                           # 2. Aplicar transforms Para normalizar 
        else:
            cropped_img = transforms.ToTensor()(cropped_img)                    # Si no hay transform, al menos convertir a Tensor para evitar el crash del DataLoader

        if not self.ssl_mode: # flujo para finetunning de clasificaciones
            label = int(row['label'])
            return cropped_img, label
        else:
            return cropped_img

    
    def get_loader(self, split_rel=0.8, seed=42, batch_size=32, num_workers=0):
        generator = torch.Generator().manual_seed(seed)                             # seed para Reproducibilidad
        train_size = int(split_rel * len(self))                                     # obtiene longitud del dataset al 80%
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split( self, [train_size, val_size], generator = generator)
        if not self.ssl_mode: # clasification original 
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
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=True)
        else:
            train_dataset, val_dataset = random_split(self, [train_size, val_size], generator = generator)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)

        # dataset debe ser una clase que herede de Dataset y lea tus rutas desde el CSV. Usa random_split o train_test_split para separar entrenamiento/validación
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
        
        return train_loader, val_loader

class MammoFeatureDataset(Dataset):
    """
    Dataset para extracción de características desde un CSV.

    Devuelve:
        image: Tensor [C, H, W]
        path: ruta completa de la imagen
    """

    def __init__(self, csv_file, transform=data_transforms, crop=False):
        self.logger = logging.getLogger(__name__)

        self.csv_file = Path(csv_file)
        self.root_dir = self.csv_file.parent
        self.data = pd.read_csv(csv_file)

        self.transform = transform
        self.crop = crop

        # Verificar que existan las columnas necesarias
        required_columns = ["preprocessed_image_path", "mask_path"]

        missing_columns = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"El CSV no contiene las columnas requeridas: {missing_columns}"
            )

        # Eliminar imágenes cuyo archivo no existe
        exists = self.data["preprocessed_image_path"].apply(
            lambda p: (
                isinstance(p, str)
                and os.path.isfile(os.path.join(self.root_dir, p))
            )
        )

        n_before = len(self.data)

        self.data = self.data[exists].reset_index(drop=True)

        n_removed = n_before - len(self.data)

        if n_removed > 0:
            self.logger.warning(
                f"{n_removed} imágenes listadas en el CSV no existen en disco; "
                f"se omiten."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        # --------------------------------------------------
        # Imagen
        # --------------------------------------------------
        img_path = os.path.join(
            self.root_dir,
            row["preprocessed_image_path"]
        )

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # --------------------------------------------------
        # Máscara / crop
        # --------------------------------------------------
        cropped = image_np

        if self.crop:

            mask_path = os.path.join(
                self.root_dir,
                row["mask_path"]
            )

            try:
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask)

                if mask.size == image.size:

                    coords = np.argwhere(mask_np > 0)

                    if len(coords) > 0:

                        y0, x0 = coords.min(axis=0)
                        y1, x1 = coords.max(axis=0) + 1

                        cropped = image_np[y0:y1, x0:x1]

                else:
                    self.logger.debug(
                        f"Máscara inválida para {img_path}. "
                        f"Usando imagen completa."
                    )

            except (FileNotFoundError, OSError):
                self.logger.debug(
                    f"Máscara no encontrada para {img_path}. "
                    f"Usando imagen completa."
                )

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------
        cropped_img = Image.fromarray(cropped)

        if self.transform is not None:
            cropped_img = self.transform(cropped_img)
        else:
            cropped_img = transforms.ToTensor()(cropped_img)

        # --------------------------------------------------
        # Devolver imagen + path
        # --------------------------------------------------
        return cropped_img, img_path

    def get_loader(self, batch_size=32, num_workers=0):

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return loader
if __name__ == "__main__":
    
    ruta_csv = Path("../../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    dataset = MammoDataset(csv_file=ruta_csv, ssl_mode = True)
    train_loader, val_loader = dataset.get_loader()
    #aqui va train y val.
