"""Dataset management for supervised density classification and SSL pretraining.

SSL pipeline (per image):
    DICOM -> float32 array -> per-image min-max to [0, 1] (MONOCHROME1 inverted)
    -> (1, H, W) tensor -> resize keeping aspect ratio + center zero-padding to
    512x512 -> random rotation by a multiple of 90 degrees -> random horizontal
    flip -> random vertical flip.

Validation uses only the deterministic resize + padding step so that the
validation loss is comparable between epochs and between Optuna trials.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms

IMAGE_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeWithPadding: # candidto a salir para tratar todo como tensor
    """Resize a PIL image while preserving aspect ratio and pad it to a square."""

    def __init__(self, size: int = IMAGE_SIZE, fill: int = 0) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        """Resize and center-pad an image to the configured square size."""
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.BILINEAR)

        pad_width = self.size - new_width
        pad_height = self.size - new_height
        # Padding order: left, top, right, bottom.
        padding = (
            pad_width // 2,
            pad_height // 2,
            pad_width - pad_width // 2,
            pad_height - pad_height // 2,
        )
        return ImageOps.expand(image, padding, fill=self.fill)


class PadToSquareTensor: # -------------C A M B I O--------------------------
    """Resize a (C, H, W) float tensor keeping aspect ratio and center square.

    The longest side becomes ``size``; the shortest side is padded with
    ``fill`` (0.0 = black). Nothing is cropped and nothing is stretched.
    Anti-aliasing is enabled because DICOM mammograms (~3000x4000) are
    downscaled by almost an order of magnitude.
    """

    def __init__(self, size: int = IMAGE_SIZE, fill: float = 0.0) -> None:
        self.size = size
        self.fill = fill

    def geometry(self, height: int, width: int) -> tuple[int, int, tuple[int, int, int, int]]:
        """Return (new_h, new_w, (left, right, top, bottom)) for an input size."""
        scale = self.size / max(height, width)
        new_h = max(1, round(height * scale))
        new_w = max(1, round(width * scale))
        pad_h = self.size - new_h
        pad_w = self.size - new_w
        pads = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        return new_h, new_w, pads

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        new_h, new_w, pads = self.geometry(height, width)
        if (new_h, new_w) != (height, width):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            image = image.clamp_(0.0, 1.0)
        # F.pad order for the last two dims: (left, right, top, bottom).
        return F.pad(image, pads, mode="constant", value=self.fill)


# Standard transformation for supervised training and validation.
data_transforms = transforms.Compose(
    [
        # Preserve aspect ratio and avoid image distortion with black padding.
        ResizeWithPadding(IMAGE_SIZE, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

# SSL: operates on one-channel float tensors in [0, 1].
ssl_train_transforms = transforms.Compose(
    [
        PadToSquareTensor(IMAGE_SIZE, fill=0.0),
        transforms.RandomRotation(degrees=(0, 90)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)
# -------------C A M B I O--------------------------
ssl_val_transforms = PadToSquareTensor(IMAGE_SIZE, fill=0.0)
ssl_transforms = ssl_train_transforms  # Backwards-compatible alias.

# -------------C A M B I O--------------------------
class _SSLSubset(Dataset):
    """View over selected SSL samples with its own transform."""

    def __init__(
        self,
        base: MammoDataset,
        indices: list[int],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.base._load_ssl_tensor(self.indices[idx]))


class MammoDataset(Dataset):
    """Load mammography data for classification or self-supervised learning."""

    class_to_idx = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }

    def __init__(
        self,
        csv_file: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = data_transforms,
        ssl_mode: bool = False,
        crop: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.root_dir = Path(csv_file).parent
        self.data = pd.read_csv(csv_file)
        self.ssl_mode = ssl_mode
        self.crop = crop

        if self.ssl_mode:
            self.transform = ssl_train_transforms
            self.val_transform = ssl_val_transforms
            self._prepare_ssl_data()
        else:
            self.transform = transform
            self._prepare_classification_data()

    def _prepare_classification_data(self) -> None:
        """Prepare and validate samples for supervised density classification."""
        self.data["label"] = self.data["density"].map(self.class_to_idx)
        self.data = self.data[
            ["preprocessed_image_path", "view", "label"]
        ].copy()

        valid_rows = (
            self.data["view"].ne("MLO")
            & self.data["view"].notna()
            & self.data["label"].notna()
        )
        self.data = self.data.loc[valid_rows].copy()

        exists = self.data["preprocessed_image_path"].map(
            lambda path: isinstance(path, str)
            and (self.root_dir / path).is_file()
        )
        self._remove_missing_files(exists, "classification")

    def _prepare_ssl_data(self) -> None:
        """Prepare and validate samples for self-supervised pretraining."""
        self.data = self.data[["study_id", "image_id"]].copy()
        valid_files: list[bool] = []
        invalid_paths: list[Path] = []

        for study_id, image_id in zip(self.data["study_id"],self.data["image_id"]):
            image_path = self._get_ssl_image_path(study_id, image_id)
            if not image_path.is_file():# validates that the file exist
                valid_files.append(False)
                continue

            try: #validates the integrity of the file
                dataset = pydicom.dcmread(image_path)
                pixels = dataset.pixel_array
                if pixels.ndim != 2:
                    raise ValueError(
                        f"Expected a 2D grayscale DICOM, got {pixels.shape}."
                    )
                valid_files.append(True)
            except Exception as exc:
                valid_files.append(False)
                if len(invalid_paths) < 10:
                    invalid_paths.append(image_path)
                self.logger.debug("Skipping unreadable DICOM %s: %s", image_path, exc)

        num_before = len(self.data)
        self.data = self.data.loc[valid_files].reset_index(drop=True)
        num_removed = num_before - len(self.data)
        if num_removed > 0:
            examples = ", ".join(str(path) for path in invalid_paths)
            suffix = f" Examples: {examples}" if examples else ""
            self.logger.warning(
                "%d invalid or missing SSL DICOMs were omitted.%s",
                num_removed,
                suffix,
            )

    def _remove_missing_files(
        self,
        exists: pd.Series | list[bool],
        dataset_name: str,
    ) -> None:
        """Discard rows whose image file does not exist on disk."""
        num_before = len(self.data)
        self.data = self.data.loc[exists].reset_index(drop=True)
        num_removed = num_before - len(self.data)
        if num_removed > 0:
            self.logger.warning(
                "%d images listed in the CSV do not exist on disk; they were "
                "omitted from the %s dataset.",
                num_removed,
                dataset_name,
            )

    def _get_ssl_image_path(self, study_id: Any, image_id: Any) -> Path:
        """Build the DICOM path for an SSL sample."""
        return self.root_dir / "images" / str(study_id) / f"{image_id}.dicom"

    @staticmethod
    def _dicom_to_tensor(dataset: pydicom.dataset.FileDataset) -> torch.Tensor:
        """Convert a DICOM image to a normalized one-channel float tensor.

        The conversion preserves the original DICOM bit depth until after
        per-image min-max normalization, avoiding 8-bit quantization.
        MONOCHROME1 images (where high values are dark) are inverted so that
        every image has the same polarity (tissue bright, background black).
        """
        pixels = dataset.pixel_array.astype(np.float32)
        if pixels.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale DICOM, got pixel_array shape {pixels.shape}."
            )

        minimum = pixels.min()
        maximum = pixels.max()
        if maximum > minimum:
            pixels = (pixels - minimum) / (maximum - minimum)
        else:
            pixels = np.zeros_like(pixels, dtype=np.float32)

        photometric = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric.upper() == "MONOCHROME1":
            pixels = 1.0 - pixels

        return torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0)
    # -------------C A M B I O--------------------------
    def _load_ssl_tensor(self, idx: int) -> torch.Tensor:
        """Read one DICOM from disk as a (1, H, W) float32 tensor in [0, 1]."""
        row = self.data.iloc[idx]
        image_path = self._get_ssl_image_path(row["study_id"], row["image_id"])
        return self._dicom_to_tensor(pydicom.dcmread(image_path))

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, int]:
        """Load and transform a single sample."""
        if self.ssl_mode:
            return self.transform(self._load_ssl_tensor(idx))

        row = self.data.iloc[idx]
        image_path = self.root_dir / row["preprocessed_image_path"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, int(row["label"])

    def get_loader(
        self,
        split_rel: float = 0.8,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 0,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Create reproducible training and validation data loaders.

        ``max_train_samples`` / ``max_val_samples`` (SSL mode only) keep a fixed
        random subset of each split; useful for fast hyperparameter search.
        """
        if not 0 < split_rel < 1:
            raise ValueError("split_rel must be between 0 and 1.")
        if len(self) == 0:
            raise ValueError("The dataset is empty after filtering invalid samples.")

        generator = torch.Generator().manual_seed(seed)
        train_size = int(split_rel * len(self))
        val_size = len(self) - train_size
        train_dataset, val_dataset = random_split(self,[train_size, val_size], generator=generator )

        pin_memory = torch.cuda.is_available()
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }

        if self.ssl_mode:
            # random_split returns a random permutation, so truncating the index
            # lists yields a reproducible random subset.
            train_idx = list(train_dataset.indices)[:max_train_samples]
            val_idx = list(val_dataset.indices)[:max_val_samples]
            # Training: random augmentations. Validation: deterministic.
            train_loader = DataLoader(
                _SSLSubset(self, train_idx, self.transform),
                shuffle=True,
                generator=generator,
                **loader_kwargs,
            )
            val_loader = DataLoader(
                _SSLSubset(self, val_idx, self.val_transform),
                shuffle=False,
                **loader_kwargs,
            )
            return train_loader, val_loader

        train_indices = train_dataset.indices
        train_labels = self.data.iloc[train_indices]["label"].to_numpy(
            dtype=np.int64
        )
        class_counts = torch.bincount(
            torch.as_tensor(train_labels, dtype=torch.long),
            minlength=len(self.class_to_idx),
        )
        class_weights = torch.zeros_like(class_counts, dtype=torch.float)
        nonzero_classes = class_counts > 0
        class_weights[nonzero_classes] = 1.0 / class_counts[
            nonzero_classes
        ].float()
        sample_weights = class_weights[torch.as_tensor(train_labels)]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs,
        )
        return train_loader, val_loader
