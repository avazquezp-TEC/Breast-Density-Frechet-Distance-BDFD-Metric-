from torch.utils.data import Dataset
import os
from PIL import Image


class SimpleImageDataset(Dataset):
    def __init__(self, image_folder, image_mode, transform=None):

        if isinstance(image_folder, (str, os.PathLike)):
            image_folders = [image_folder]
        else:
            image_folders = image_folder

        self.image_paths = []

        for folder in image_folders:
            self.image_paths.extend(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))
            )

        self.transform = transform
        self.image_mode = image_mode

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path).convert(self.image_mode)

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.image_paths)