"""
Extiende `ood_detection.py` de mazurowski-lab/medical-image-similarity-metrics
agregando tu propio ConvNeXt V2 (SSL + fine-tuned en MammoBench) como una
representación de características más, evaluada con EXACTAMENTE el mismo
protocolo que usan para ImageNet/RadImageNet/Radiomics en el paper de FRD:

    1. Centroide del set de referencia in-distribution (R1).
    2. Score por imagen = distancia L2 de sus features a ese centroide
       (nota: es L2 al centroide, NO Mahalanobis (me gustaria).
    3. AUC (¿el score separa ID de OOD?) + umbral del percentil 95 de los
       scores del propio set de referencia -> accuracy/TPR/TNR.

ESTE ARCHIVO VA DENTRO DEL CLON de medical-image-similarity-metrics
https://github.com/mazurowski-lab/medical-image-similarity-metrics.git
(mismo nivel que ood_detection.py), porque reutiliza sus módulos
`src.utils` / `src.dataset` / `ood_detection.get_imagenet_features`.

Para poder importar el extractor (c_dataset_manage.py,
d_frechet_metrics.py), pásale --project-dir apuntando a la carpeta donde
viven esos archivos.

Uso:
    python e_ood_detection_convnextv2.py \
        --img_folder_ref_id R1/ \
        --img_folder_test_id R2/ \
        --img_folder_test_ood S1_o_anomalas/ \
        --checkpoint /ruta/convnextv2_tiny_custom.pth \
        --project-dir /ruta/a/tu/proyecto/mammo

    # Para iterar rápido probando solo Breast Density Frechet Distance (sin ImageNet/
    # RadImageNet/Radiomics, que son mucho más lentas
    python e_ood_detection_convnextv2.py ... --only ConvNeXtV2-Mammo
    python e_ood_detection_convnextv2.py --img_folder_ref_id ~/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Data --img_folder_test_id ~/model_test/mammography_validation/Val_set --img_folder_test_ood ~/model_test/mammography_validation/Grayscale_CC
    python medical_image_similarity_metrics/e_ood_detection_convnextv2.py --img_folder_ref_id  ~/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/mammo-bench.csv --img_folder_test_id ~/model_test/mammography_validation/Val_set --img_folder_test_ood ~/model_test/mammography_validation/Grayscale_CC --checkpoint ~/model_test/mammography_validation/Breast-Density-Frechet-Distance-BDFD-Metric-/convnextv2_tiny_custom.pth --project-dir ~/model_test/mammography_validation/Breast-Density-Frechet-Distance-BDFD-Metric-
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import scipy.stats as sps
import torch
from sklearn.metrics import roc_auc_score

# --- Reutiliza el framework original tal cual está ---
from ood_detection import get_imagenet_features, auc_deviation
from src.radiomics.radiomics_utils import (
    convert_radiomic_dfs_to_vectors,
    compute_and_save_imagefolder_radiomics_parallel,
)
from c_dataset_manage import MammoFeatureDataset, data_transforms


fix_seed = True
if fix_seed:
    seed = 1338
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# ConvNeXt V2 propio como representación de features (misma firma de salida
# que get_imagenet_features en ood_detection.py)
# --------------------------------------------------------------------------

def get_convnextv2_features(in_img_folder, out_img_folder_id, out_img_folder_ood, checkpoint, num_classes=4, bs=16):
    """Extrae features con tu propio ConvNeXt V2 (d_frechet_metrics.py),
    devolviendo el mismo formato que get_imagenet_features()."""
    from d_frechet_metrics import ImageFolderDataset, extract_features, get_device, load_feature_extractor
    from torch.utils.data import DataLoader

    device = get_device()
    model = load_feature_extractor(checkpoint, num_classes=num_classes, device=device)

    # ruta_csv = Path("../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
    if in_img_folder.endswith(".csv"):
        print("La ruta apunta a un CSV")
        in_dataset = MammoFeatureDataset(csv_file=in_img_folder)

        in_loader = in_dataset.get_loader(
            batch_size=bs
        )                             # crea el objeto MammoDataset que creara los loaders de entrenamiento y validacion
    else:
        in_loader = DataLoader(ImageFolderDataset(in_img_folder), batch_size=bs, shuffle=False)
    
    out_loader_id = DataLoader(ImageFolderDataset(out_img_folder_id), batch_size=bs, shuffle=False)
    out_loader_ood = DataLoader(ImageFolderDataset(out_img_folder_ood), batch_size=bs, shuffle=False)

    in_feats, _ = extract_features(model, in_loader, device)
    id_feats, _ = extract_features(model, out_loader_id, device)
    ood_feats, _ = extract_features(model, out_loader_ood, device)

    return torch.tensor(in_feats), torch.tensor(id_feats), torch.tensor(ood_feats)


# --------------------------------------------------------------------------
# Lógica de evaluación de ood_detection.py::main(), factorizada para no
# repetirla 4 veces (una por representación)
# --------------------------------------------------------------------------

def evaluate_representation(in_activations, out_activations_id, out_activations_ood, val_frac=0.1, use_val_set=False):
    in_activations = np.asarray(in_activations, dtype=np.float64)
    out_activations_id = torch.as_tensor(np.asarray(out_activations_id, dtype=np.float64))
    out_activations_ood = torch.as_tensor(np.asarray(out_activations_ood, dtype=np.float64))

    val_idx = np.random.choice(in_activations.shape[0], max(1, int(val_frac * in_activations.shape[0])), replace=False)
    train_idx = np.array([i for i in range(in_activations.shape[0]) if i not in val_idx])

    in_activations_val = torch.as_tensor(in_activations[val_idx])
    in_activations_train = torch.as_tensor(in_activations[train_idx])
    if not use_val_set:
        in_activations_val = in_activations_train
        print("(usando split del propio set de referencia como validación, no un holdout real)")

    id_mean = in_activations_train.mean(dim=0)

    labels = np.concatenate([np.zeros(len(out_activations_id)), np.ones(len(out_activations_ood))])
    scores = torch.stack([torch.norm(id_mean - out, dim=0) for out in torch.cat([out_activations_id, out_activations_ood])])
    scores = scores.detach().numpy()

    auc = roc_auc_score(labels, scores)

    ID_scores = scores[: len(out_activations_id)]
    ID_scores_val = torch.stack([torch.norm(id_mean - out, dim=0) for out in in_activations_val]).detach().numpy()
    threshold = np.percentile(ID_scores_val, 95)

    pred = scores > threshold
    acc = np.mean(pred == labels)
    tpr = np.sum((pred == 1) & (labels == 1)) / max(np.sum(labels == 1), 1)
    fpr = np.sum((pred == 1) & (labels == 0)) / max(np.sum(labels == 0), 1)
    tnr = 1 - fpr

    return {"auc": auc, "threshold": float(threshold), "accuracy": acc, "tpr": tpr, "tnr": tnr}


# --------------------------------------------------------------------------
# Orquestación: mismas 3 representaciones originales + ConvNeXtV2-Mammo
# --------------------------------------------------------------------------

def main(
    in_img_folder,
    out_img_folder_id,
    out_img_folder_ood,
    checkpoint,
    num_classes=4,
    val_frac=0.1,
    use_val_set=False,
    only=None,
):
    all_names = ["Radiomics", "ConvNeXtV2-Mammo", "RadImageNet", "ImageNet"] #,
    feature_names = [n for n in all_names if (only is None or n in only)]

    results = {}
    for features_name in feature_names:
        print(f"\n=== Resultados usando features de {features_name} ===")
        in_img_folder2 = [
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/cdd-cesm",
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/cmmd",
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/ddsm",
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/dmid",
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/inbreast",
                        "/home/angel/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/Preprocessed_Dataset/kau-bcmd"]
        if "ImageNet" in features_name:
            
            in_act, id_act, ood_act = get_imagenet_features(in_img_folder2, out_img_folder_id, out_img_folder_ood, features_name)

        elif features_name == "Radiomics":
            # IN-distribution: six Mammo-Bench folders
            radiomics_dfs_in = []

            for folder in in_img_folder2:

                radiomics_path = os.path.join(folder, "radiomics.csv")

                if not os.path.exists(radiomics_path):
                    print(f"Calculando radiomics para {folder} ...")
                    compute_and_save_imagefolder_radiomics_parallel(folder)

                radiomics_dfs_in.append(pd.read_csv(radiomics_path))

            # Combine the six Mammo-Bench datasets
            radiomics_df1 = pd.concat(
                radiomics_dfs_in,
                ignore_index=True
            )

            # ID
            radiomics_path2_id = os.path.join(out_img_folder_id, "radiomics.csv")

            if not os.path.exists(radiomics_path2_id):
                print(f"Calculando radiomics para {out_img_folder_id} ...")
                compute_and_save_imagefolder_radiomics_parallel(out_img_folder_id)

            radiomics_df2_id = pd.read_csv(radiomics_path2_id)

            # OOD
            radiomics_path2_ood = os.path.join(out_img_folder_ood, "radiomics.csv")

            if not os.path.exists(radiomics_path2_ood):
                print(f"Calculando radiomics para {out_img_folder_ood} ...")
                compute_and_save_imagefolder_radiomics_parallel(out_img_folder_ood)
                # compute_and_save_imagefolder_radiomics(out_img_folder_ood)

            radiomics_df2_ood = pd.read_csv(radiomics_path2_ood)

            # Convert to feature vectors
            in_act, id_act, _, _ = convert_radiomic_dfs_to_vectors(
                radiomics_df1,
                radiomics_df2_id,
                match_sample_count=True,
                return_image_fnames=True
            )

            _, ood_act, _, _ = convert_radiomic_dfs_to_vectors(
                radiomics_df1,
                radiomics_df2_ood,
                match_sample_count=True,
                return_image_fnames=True
            )

        elif features_name == "ConvNeXtV2-Mammo":
            if checkpoint is None:
                print("Sin --checkpoint, se omite ConvNeXtV2-Mammo.")
                continue
            in_act, id_act, ood_act = get_convnextv2_features(
                in_img_folder, out_img_folder_id, out_img_folder_ood, checkpoint, num_classes=num_classes
            )

        result = evaluate_representation(in_act, id_act, ood_act, val_frac=val_frac, use_val_set=use_val_set)
        results[features_name] = result
        print(
            f"AUC={result['auc']:.3f} | Accuracy={result['accuracy']:.3f} | "
            f"TPR(sensibilidad)={result['tpr']:.3f} | TNR(especificidad)={result['tnr']:.3f} | "
            f"umbral(p95)={result['threshold']:.4f}"
        )

    print("\n=== Resumen ===")
    summary_df = pd.DataFrame(results).T
    print(summary_df.round(3).to_string())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOD detection con ImageNet/RadImageNet/Radiomics/ConvNeXtV2-Mammo.")
    parser.add_argument("--img_folder_ref_id", type=str, required=True, help="R1: carpeta de referencia in-distribution")
    parser.add_argument("--img_folder_test_id", type=str, required=True, help="R2: carpeta de prueba in-distribution")
    parser.add_argument("--img_folder_test_ood", type=str, required=True, help="S1 o anómalas: carpeta out-of-distribution")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta al .pth de c_train_convnext.py (para ConvNeXtV2-Mammo)")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--project-dir", type=str, required=True, help="Carpeta con c_dataset_manage.py / d_frechet_metrics.py")
    parser.add_argument(
        "--only", type=str, default=None,
        help="Lista separada por comas de representaciones a evaluar, p.ej. 'ConvNeXtV2-Mammo,Radiomics'. Por defecto, las 4.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--use-val-set", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, args.project_dir)

    only = [s.strip() for s in args.only.split(",")] if args.only else None

    main(
        args.img_folder_ref_id,
        args.img_folder_test_id,
        args.img_folder_test_ood,
        args.checkpoint,
        num_classes=args.num_classes,
        val_frac=args.val_frac,
        use_val_set=args.use_val_set,
        only=only,
    )
