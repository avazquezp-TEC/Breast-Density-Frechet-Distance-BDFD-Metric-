# Breast Density Fréchet Distance (BDFD) Metric

> **Experimental research code.** This repository is under active development and is not validated for clinical use or diagnostic decision-making.

This project develops and evaluates deep-learning workflows for breast-density classification and for assessing the statistical similarity between real and generated mammography images. The long-term goal is to support privacy-aware research pipelines—including federated learning, differential privacy, and synthetic-image generation—while providing quantitative tools to evaluate image quality and detect distributional anomalies.

## Current scope

The repository currently includes experiments for:

- Fine-tuning **ConvNeXt-Tiny** for four-class breast-density classification.
- Self-supervised learning (SSL) with a masked-autoencoder-style workflow based on **ConvNeXt V2**.
- Extracting backbone features for distribution-comparison metrics, including Fréchet-based evaluation.
- Out-of-distribution (OOD) detection using radiomics and deep feature representations.

The code, hyperparameters, datasets, and results may change as experiments are refined. Reported metrics should be interpreted as internal experimental results, not as benchmark claims.

## Repository structure

| Script | Purpose |
|---|---|
| `a_prueba_convnext.py` | Runs inference with a ConvNeXt-Tiny model. |
| `b_train_convnext.py` | Fine-tunes ConvNeXt-Tiny for four-class breast-density classification. |
| `c_dataset_manage.py` | Dataset utilities adapted for the SSL workflow. |
| `c_train_convnext.py` | Fine-tunes the classifier after SSL pretraining. |
| `c_train_mae_convnextv2.py` | SSL pretraining with masking using the remaining unlabeled dataset images. |
| `d_frechet_metrics.py` | Computes Fréchet-based metrics between real and generated image sets. |
| `e_ood_detection_convnextv2.py` | Evaluates OOD-detection experiments. |

## Environment

Activate the appropriate Python environment before running the scripts.

### macOS

```bash
source ~/mac_metal311/bin/activate
```

### Linux

A separate Python 3.11 environment is currently recommended for the PyTorch workflow:

```bash
python3.11 -m venv ~/torch_venv
source ~/torch_venv/bin/activate
```

> The commands above reflect the current development setup. They may need to be adapted to your local Python, CUDA, PyTorch, and dependency configuration.

Several scripts expect a dataset CSV path. Update it for your local environment, for example:

```python
ruta_csv = Path("../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
```

## Usage

### A. ConvNeXt-Tiny inference

Run inference on a single image:

```bash
python a_prueba_convnext.py --image ../../../Sergio/SRIR/DATASET/DIV2K_train_HR/0189.png
```

### B. Supervised ConvNeXt-Tiny training

The initial training workflow replaces the original 1,000-class classification head with a four-class breast-density head and fine-tunes the backbone.

```bash
python b_train_convnext.py
```

The current implementation includes:

- `WeightedRandomSampler` to mitigate class imbalance.
- `ReduceLROnPlateau` for learning-rate scheduling.
- Transfer learning with `ConvNeXt_Tiny_Weights.DEFAULT`.
- ImageNet preprocessing.
- Early stopping (initially configured with patience of 7; this criterion remains under review).

### C. Self-supervised learning and fine-tuning

The SSL workflow uses `c_train_mae_convnextv2.py` to pretrain on the remaining 12,521 images using masked-image reconstruction. The resulting representation can then be fine-tuned using `c_train_convnext.py`.

Supported arguments include:

```text
--csv         Path to the dataset CSV file
--epochs      Maximum number of training epochs
--patience    Early-stopping patience
--batch-size  Batch size
--lr          Learning rate
--mask-ratio  Masking ratio
--out         Output directory
```

Example experimental configuration for SSL pretraining:

- Batch size: 32
- Learning rate: $1\times10^{-4}$ with `CosineAnnealingLR`
- Mask ratio: 0.7
- Maximum epochs: 200
- Early-stopping patience: 14
- Minimum delta: $1\times10^{-4}$
- Random seed: 42

The reported SSL run trained for 105 epochs before restoring the model selected by the early-stopping procedure. Fine-tuning was configured with a batch size of 16, a backbone learning rate of $1\times10^{-5}$, a classification-head learning rate of $1\times10^{-4}$, `ReduceLROnPlateau`, and random seed 42.

### D. Fréchet-based comparison of real and generated images

Use the following command pattern to compare real and generated images:

```bash
python d_frechet_metrics.py \
  --original-path /path/to/real_images \
  --artificial-path /path/to/generated_images \
  --checkpoint convnextv2_tiny_custom.pth \
  --out-dir frechet_results/
```

This module is intended to support experiments with Fréchet distance, multidimensional-scaling projections, and alternative kernel-based metrics such as MMD. Its statistical validity and sensitivity are still being evaluated.

## Experimental results

### Supervised baseline

The following validation result was obtained with the supervised ConvNeXt-Tiny configuration:

```text
Loss: 0.8946
Accuracy: 60.26%
Macro precision: 59.65%
Macro recall: 62.75%
Macro F1: 60.42%
```

### Comparison before and after SSL

A separate baseline run reported:

```text
Loss: 0.9627
Accuracy: 56.66%
Macro precision: 58.91%
Macro recall: 60.17%
Macro F1: 56.83%
```

After SSL pretraining and fine-tuning, the reported validation result was:

```text
Loss: 0.8177
Accuracy: 64.65%
Macro precision: 64.61%
Macro recall: 67.47%
Macro F1: 65.18%
```

Reported confusion matrix for the SSL experiment (rows: ground truth; columns: prediction; class order: A, B, C, D):

```text
[[151,  59,  10,   8],
 [ 77, 325, 167,  18],
 [  5,  35, 280,  82],
 [  4,   6,  37, 173]]
```

These preliminary results suggest improved validation performance after SSL pretraining. However, they require replication with fixed data splits, documented preprocessing, confidence intervals, and external test sets before drawing comparative conclusions.

## OOD detection experiment
Para ejecutar este programa es necesario clonar el repositorio
https://github.com/mazurowski-lab/medical-image-similarity-metrics.git
Se han modificado algunos archivos `medial_image_similarity_metrics/src/radiomics/radiomics_utils.py`, `medial_image_similarity_metrics/src/dataset.py`y `medial_image_similarity_metrics/ood_detection.py``
hay que instalar radiomics
`pip install git+https://github.com/AIM-Harvard/pyradiomics.git`

y copiar los archivos `e_dataset_manage.py`y `e_ood_detection_convnextv2.py` en la raiz de `medial_image_similarity_metrics`.

ejecutar con los siguientes args:
```bash
e_ood_detection_convnextv2.py
"args": [
                "--img_folder_ref_id",  "~/model_test/mammography_validation/Experimentos/Mammo_Bench_v2/mammo-bench.csv",
                "--img_folder_test_id", "~/model_test/mammography_validation/Val_set",
                "--img_folder_test_ood", "~/model_test/mammography_validation/Grayscale_CC",
                "--checkpoint", "~/model_test/mammography_validation/Breast-Density-Frechet-Distance-BDFD-Metric-/convnextv2_tiny_custom.pth",
                "--project-dir", "~/model_test/mammography_validation/Breast-Density-Frechet-Distance-BDFD-Metric-"
            ]
```

The initial OOD experiment compares images generated by Jorge against MexBreast, using MammoBench as the reference distribution. The current reported results are:

| Method | AUC | Threshold | Accuracy | TPR | TNR |
|---|---:|---:|---:|---:|---:|
| Radiomics | 0.997 | 33.906 | 0.897 | 1.000 | 0.806 |
| ConvNeXt V2-Tiny | 0.986 | 22.316 | 0.925 | 0.862 | 0.988 |
| RadImageNet | 0.390 | 0.816 | 0.817 | 0.039 | 0.799 |
| ImageNet | 0.726 | 17.551 | 0.503 | 0.062 | 0.948 |

These values are exploratory and should not be interpreted as final comparisons. The evaluation protocol, thresholds, and dataset partitions should be fully documented before publication or external use.

## Metrics

For multiclass classification, the repository reports macro-averaged precision, recall, and F1 score:

- **Precision:** $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ — among samples predicted as a class, the proportion that truly belong to that class.
- **Recall:** $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ — among samples that truly belong to a class, the proportion correctly identified.
- **F1 score:** harmonic mean of precision and recall.

A confusion matrix is read with rows as true labels and columns as predicted labels. Larger values on the main diagonal indicate more correct predictions; off-diagonal values represent class confusions.

## Limitations

- This repository is experimental and may contain incomplete or changing scripts.
- Dataset paths are currently environment-specific.
- Reproducibility artifacts such as pinned dependencies, automated tests, standardized splits, and experiment tracking are still being consolidated.
- The models and metrics have not been clinically validated.
- Results may be sensitive to class imbalance, data-source shifts, and preprocessing choices.

## Planned work

- Evaluate hybrid architectures, including Swin Transformer–based approaches.
- Incorporate Dr. Vin mammography data for SSL pretraining, subject to the applicable governance and access requirements.
- Use MammoBench for the classification stage.
- Evaluate generalization on MammoMX and BreastMex, along with additional external datasets where appropriate.
- Increase the number of images used for Fréchet-based evaluation. The current collection contains approximately 1,024 images; larger sample sizes are needed for more stable estimates, with a target of at least 8,192 images under consideration.
- Add reproducible configuration files, dependency specifications, and a formal evaluation protocol.

## Responsible use

This software is provided for research and development only. It must not be used as a medical device, for clinical diagnosis, or to make patient-care decisions without appropriate validation, regulatory review, and institutional approval.
