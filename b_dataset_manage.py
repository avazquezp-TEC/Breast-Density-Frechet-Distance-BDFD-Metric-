import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms

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
            print("mascara invalida. saltando...")
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


if __name__ == "__main__":
    
    ruta_csv = Path("../../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    dataset = MammoDataset(csv_file=ruta_csv, transform=data_transforms)
    train_loader, val_loader = dataset.get_loader()
    #aqui va train y val.

    

