"""
Distancias distribucionales (Fréchet, KID/MMD) entre un conjunto de
imágenes "original" y uno "artificial", usando como extractor de
características el ConvNeXt V2 ya entrenado (checkpoint de
c_train_convnext.py) SIN su capa de clasificación.

Todo está implementado a mano (no se usa `pytorch-fid`, `torchmetrics`
ni ninguna función "calculate_fid" empaquetada) para poder:
    1. Calcular Fréchet Distance (asume gaussianidad, vía media+covarianza).
    2. Calcular KID / MMD con kernels intercambiables (polinomial y RBF),
       que NO asumen gaussianidad y complementan a Fréchet.
    3. Proyectar la nube de puntos (original + artificial) con MDS para
       inspección visual.
    4. Marcar, por imagen, qué tan lejos está de la distribución original
       (distancia de Mahalanobis), para señalar anómalas.

Uso:
    python d_frechet_metrics.py \
        --original-path /ruta/a/imagenes_reales \
        --artificial-path /ruta/a/imagenes_generadas \
        --checkpoint convnextv2_tiny_custom.pth \
        --out-dir resultados_frechet/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display, solo guardar a archivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from sklearn.manifold import MDS
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from c_dataset_manage import data_transforms

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# 1. Extractor de características: ConvNeXt V2 sin la capa de clasificación
# --------------------------------------------------------------------------

def load_feature_extractor(
    checkpoint_path: str | Path,
    num_classes: int = 4,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Reconstruye EXACTAMENTE la arquitectura de c_train_convnext.py
    (ConvNeXt V2 tiny + head.fc = Dropout(0.3) + Linear), carga los pesos
    ya afinados, y luego reemplaza head.fc por Identity para dejar el
    modelo como extractor de 768 características. Modo eval + inference.
    """
    device = device or get_device()

    model = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=num_classes)
    in_features = model.head.fc.in_features
    model.head.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_classes))

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model.head.fc = nn.Identity()  # <- quita la capa de clasificación
    model.eval()
    return model.to(device)


# --------------------------------------------------------------------------
# 2. Dataset simple: todas las imágenes de una carpeta, transform básico
# --------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    """
    Carga recursivamente todas las imágenes de `root`, aplica el
    transform básico (ResizeWithPadding + Normalize, SIN recorte por
    máscara) y devuelve (tensor, ruta) para poder rastrear qué vector
    corresponde a qué archivo.
    """
    def __init__(self, root: str | Path, transform=data_transforms):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
        if len(self.paths) == 0:
            raise RuntimeError(f"No se encontraron imágenes en {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, str(path)


@torch.inference_mode()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device):
    all_feats, all_paths = [], []
    for images, paths in tqdm(loader, desc="Extrayendo características", unit="batch"):
        images = images.to(device)
        feats = model(images)  # (B, 768) porque head.fc = Identity
        all_feats.append(feats.cpu().numpy())
        all_paths.extend(paths)
    return np.concatenate(all_feats, axis=0).astype(np.float64), all_paths


# --------------------------------------------------------------------------
# 3. Fréchet Distance (implementada a mano)
# --------------------------------------------------------------------------

def compute_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6
) -> float:
    """
    d^2 = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*sqrt(S1 @ S2))
    Si S1@S2 sale mal condicionada (covarianza singular por tener menos
    imágenes que dimensiones: 768), se regulariza sumando eps*I antes de
    la raíz matricial, igual que hacen las implementaciones de
    referencia de FID.
    """
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if not np.isfinite(covmean).all():
        logger.warning(
            "sqrtm(S1 @ S2) no es finita (covarianza probablemente singular); "
            f"regularizando con eps={eps}."
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# --------------------------------------------------------------------------
# 4. KID / MMD con kernel intercambiable (polinomial y RBF)
# --------------------------------------------------------------------------

def polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: float | None = None, coef0: float = 1.0) -> np.ndarray:
    """Kernel usado por KID en el paper original (Bińkowski et al. 2018)."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X @ Y.T + coef0) ** degree


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Kernel gaussiano, alternativa a KID que no asume la forma polinomial."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    x_sq = np.sum(X**2, axis=1)[:, None]
    y_sq = np.sum(Y**2, axis=1)[None, :]
    sq_dists = np.clip(x_sq + y_sq - 2 * X @ Y.T, a_min=0, a_max=None)
    return np.exp(-gamma * sq_dists)


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, kernel_fn, **kernel_kwargs) -> float:
    """
    Estimador insesgado (U-statistic) de MMD^2:
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    excluyendo la diagonal de Kxx/Kyy (i != i') para no sesgar hacia 0.
    Con kernel polinomial (degree=3) esto ES la métrica KID estándar.
    """
    m, n = X.shape[0], Y.shape[0]
    Kxx = kernel_fn(X, X, **kernel_kwargs)
    Kyy = kernel_fn(Y, Y, **kernel_kwargs)
    Kxy = kernel_fn(X, Y, **kernel_kwargs)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    term_xx = Kxx.sum() / (m * (m - 1))
    term_yy = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.sum() / (m * n)
    return float(term_xx + term_yy - 2 * term_xy)


# --------------------------------------------------------------------------
# 5. Anomalías: distancia de Mahalanobis de cada imagen a la distribución
#    "original" (mu_orig, sigma_orig)
# --------------------------------------------------------------------------

def mahalanobis_distances(features: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    d(x) = sqrt((x - mu_ref)^T @ Sigma_ref^-1 @ (x - mu_ref))
    Se usa pseudo-inversa (no inversa directa) porque con 768
    dimensiones y pocas imágenes la covarianza puede ser singular.
    """
    d = sigma_ref.shape[0]
    inv_sigma = np.linalg.pinv(sigma_ref + eps * np.eye(d))
    diff = features - mu_ref
    m2 = np.einsum("ij,jk,ik->i", diff, inv_sigma, diff)
    return np.sqrt(np.clip(m2, a_min=0, a_max=None))


def flag_anomalies(feats_orig: np.ndarray, feats_art: np.ndarray, mu_orig: np.ndarray, sigma_orig: np.ndarray, percentile: float = 95.0):
    """
    Calibra el umbral con la distancia de Mahalanobis del propio set
    original contra sí mismo (qué tan lejos de su centro está una imagen
    real "típica"), y marca como anómala cualquier imagen artificial que
    supere ese percentil.
    """
    d_orig_self = mahalanobis_distances(feats_orig, mu_orig, sigma_orig)
    threshold = float(np.percentile(d_orig_self, percentile))
    d_art = mahalanobis_distances(feats_art, mu_orig, sigma_orig)
    is_anomaly = d_art > threshold
    return d_art, is_anomaly, threshold


# --------------------------------------------------------------------------
# 6. Proyección MDS + gráfico
# --------------------------------------------------------------------------

def project_mds(feats_orig: np.ndarray, feats_art: np.ndarray, random_state: int = 42):
    X = np.concatenate([feats_orig, feats_art], axis=0)
    mds = MDS(n_components=2, metric="euclidean", n_init=4, init="random", random_state=random_state)
    coords = mds.fit_transform(X)
    n_orig = feats_orig.shape[0]
    return coords[:n_orig], coords[n_orig:]


def plot_mds(coords_orig: np.ndarray, coords_art: np.ndarray, is_anomaly: np.ndarray, out_path: str | Path):
    plt.figure(figsize=(8, 8))
    plt.scatter(coords_orig[:, 0], coords_orig[:, 1], c="tab:blue", alpha=0.6, s=20, label="Original")
    plt.scatter(coords_art[:, 0], coords_art[:, 1], c="tab:orange", alpha=0.6, s=20, label="Artificial")
    if is_anomaly.any():
        plt.scatter(
            coords_art[is_anomaly, 0], coords_art[is_anomaly, 1],
            facecolors="none", edgecolors="red", s=90, linewidths=1.5, label="Artificial anómala",
        )
    plt.title("Proyección MDS de características ConvNeXt V2")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# 7. Orquestación
# --------------------------------------------------------------------------

def main(
    original_path: str,
    artificial_path: str,
    checkpoint: str,
    num_classes: int = 4,
    batch_size: int = 16,
    num_workers: int = 0,
    out_dir: str = "resultados_frechet",
    anomaly_percentile: float = 95.0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    model = load_feature_extractor(checkpoint, num_classes=num_classes, device=device)

    orig_loader = DataLoader(ImageFolderDataset(original_path), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    art_loader = DataLoader(ImageFolderDataset(artificial_path), batch_size=batch_size, num_workers=num_workers, shuffle=False)

    feats_orig, paths_orig = extract_features(model, orig_loader, device)
    feats_art, paths_art = extract_features(model, art_loader, device)
    n_dim = feats_orig.shape[1]
    logger.info(f"Original: {feats_orig.shape[0]} imágenes | Artificial: {feats_art.shape[0]} imágenes | dim={n_dim}")

    if feats_orig.shape[0] < n_dim or feats_art.shape[0] < n_dim:
        logger.warning(
            f"Alguno de los dos conjuntos tiene menos imágenes que dimensiones ({n_dim}). "
            "La covarianza empírica será singular; Fréchet/Mahalanobis usan una "
            "regularización (eps) / pseudo-inversa, pero los resultados serán menos "
            "confiables. Idealmente se necesitan más imágenes que dimensiones."
        )

    mu_orig, sigma_orig = compute_stats(feats_orig)
    mu_art, sigma_art = compute_stats(feats_art)

    fid = frechet_distance(mu_orig, sigma_orig, mu_art, sigma_art)
    kid_poly = mmd2_unbiased(feats_orig, feats_art, polynomial_kernel, degree=3, coef0=1.0)
    mmd_rbf = mmd2_unbiased(feats_orig, feats_art, rbf_kernel)

    logger.info(f"Fréchet Distance:        {fid:.4f}")
    logger.info(f"KID (kernel polinomial): {kid_poly:.6f}")
    logger.info(f"MMD (kernel RBF):        {mmd_rbf:.6f}")

    d_art, is_anomaly, threshold = flag_anomalies(feats_orig, feats_art, mu_orig, sigma_orig, percentile=anomaly_percentile)
    logger.info(
        f"{int(is_anomaly.sum())}/{len(is_anomaly)} imágenes artificiales superan el "
        f"percentil {anomaly_percentile:.0f} de distancia de Mahalanobis del set original "
        f"(umbral={threshold:.4f})."
    )

    scores_df = pd.DataFrame({
        "path": paths_art,
        "mahalanobis_dist": d_art,
        "is_anomaly": is_anomaly,
    }).sort_values("mahalanobis_dist", ascending=False)
    scores_csv = out_dir / "anomaly_scores.csv"
    scores_df.to_csv(scores_csv, index=False)
    logger.info(f"Puntajes de anomalía guardados en: {scores_csv}")

    coords_orig, coords_art = project_mds(feats_orig, feats_art)
    mds_png = out_dir / "mds_plot.png"
    plot_mds(coords_orig, coords_art, is_anomaly, out_path=mds_png)
    logger.info(f"Proyección MDS guardada en: {mds_png}")

    summary = {
        "n_original": int(feats_orig.shape[0]),
        "n_artificial": int(feats_art.shape[0]),
        "feature_dim": int(n_dim),
        "frechet_distance": fid,
        "kid_polynomial": kid_poly,
        "mmd_rbf": mmd_rbf,
        "anomaly_percentile": anomaly_percentile,
        "mahalanobis_threshold": threshold,
        "n_anomalies": int(is_anomaly.sum()),
    }
    summary_json = out_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Resumen guardado en: {summary_json}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="FID/KID/MMD + detección de anómalas entre imágenes originales y artificiales.")
    parser.add_argument("--original-path", type=str, required=True)
    parser.add_argument("--artificial-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al .pth de c_train_convnext.py (modelo ya afinado).")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="resultados_frechet")
    parser.add_argument("--anomaly-percentile", type=float, default=95.0)
    args = parser.parse_args()

    main(
        original_path=args.original_path,
        artificial_path=args.artificial_path,
        checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        anomaly_percentile=args.anomaly_percentile,
    )
