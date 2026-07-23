# source ~/mac_metal311/bin/activate
"""
Run inference with ConvNeXt-Tiny using pretrained ImageNet-1K weights.
you can check the cathegories here
https://www.image-net.org/challenges/LSVRC/2012/browse-synsets.php
Example:
    python infer_convnext.py --image /path/to/image.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ConvNeXt-Tiny inference on a single image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    return parser.parse_args()

def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_image(image_path: Path) -> Image.Image:
    """Load an image from disk and convert it to RGB.
    If the directory is not an image, the program stops.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise IsADirectoryError(f"Expected a file, got a directory: {image_path}")

    return Image.open(image_path).convert("RGB")


def main() -> int:
    """Run inference and print the predicted class."""
    args = parse_args()
    device = get_device()
    LOGGER.info("Using device: %s", device)

    # 2) Load pre-trained weights
    weights = ConvNeXt_Tiny_Weights.DEFAULT                             # trained on IMAGENET1K_V1 with 1.2M images and 1000 classes.
    # 3) Move the device model and configure inference mode
    model = convnext_tiny(weights=weights).to(device)
    model.eval()                                                        # Equivalent to model.train(False), that is, it doesn't train, it only makes inferences
    # 4) Load Preprocessing performed during training
    preprocess = weights.transforms()
    # 5) Cargar una imagen
    image = load_image(args.image)
    inputs = preprocess(image).unsqueeze(0).to(device)                  # with preprocess(img) resulting in a tensor (C,H,W).
                                                                        # unsqueeze(0) adds another dimension, which is for the batch. Now x is (B,C,H,W) -> (1,3,224,224) according to preprocess
    with torch.inference_mode():
        logits = model(inputs)                                          #Here the inference is made. The result is called logits, which are scores produced by the last layer of the network.
        predicted_idx = logits.argmax(dim=1).item()                     # Since logits is a vector, only the highest value corresponding to the image classification is searched for here.

    categories = weights.meta["categories"]
    predicted_label = categories[predicted_idx]

    LOGGER.info("Predicted class index: %d", predicted_idx)
    LOGGER.info("Predicted label: %s", predicted_label)

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    raise SystemExit(main())
