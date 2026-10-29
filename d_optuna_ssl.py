"""Optuna study to tune SimMIM pretraining of ConvNeXt V2 on MammoBench.

Each trial trains for a few epochs on a fixed random subset of the data and is
scored with the masked-reconstruction loss on a fixed validation subset.

Important: the validation protocol is identical for every trial (same masks,
same mask ratio of ``--val-mask-ratio``, same deterministic pad-only transform).
Otherwise trials with a low training mask ratio would look "better" just
because their reconstruction task is easier.

Usage:
    python c_optuna_ssl.py --csv path/to/mammo-bench.csv --n-trials 40
    python c_train_mae_convnextv2.py --params-json best_params.json --epochs 200
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import optuna
import torch

from c_dataset_manage import IMAGE_SIZE, MammoDataset
from c_train_mae_convnextv2 import (
    ConvNeXtV2SimMIM,
    SimMIMMaskGenerator,
    fit,
    get_device,
    seed_everything,
)

logger = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial, batch_sizes: list[int]) -> dict:
    """Search space. Keys match the CLI argument names of c_train_mae_convnextv2."""
    params = {
        "lr": trial.suggest_float("lr", 2e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "mask_ratio": trial.suggest_float("mask_ratio", 0.4, 0.85, step=0.05),
        "mask_patch_size": trial.suggest_categorical("mask_patch_size", [32, 64]),
        "scheduler": trial.suggest_categorical(
            "scheduler", ["cosine", "linear", "onecycle", "plateau"]
        ),
        "drop_path": trial.suggest_float("drop_path", 0.0, 0.3, step=0.05),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.0, 1.0, 3.0]),
        "batch_size": trial.suggest_categorical("batch_size", batch_sizes),
    }
    # Warmup only makes sense for cosine/linear (onecycle has its own ramp).
    if params["scheduler"] in ("cosine", "linear"):
        params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 3)
    else:
        params["warmup_epochs"] = 0
    return params


def make_objective(dataset: MammoDataset, args: argparse.Namespace, device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.batch_sizes)
        logger.info(f"Trial {trial.number}: {params}")

        seed_everything(args.seed)  # Same split, init and shuffling for every trial.
        train_loader, val_loader = dataset.get_loader(
            seed=args.seed,
            batch_size=params["batch_size"],
            num_workers=args.num_workers,
            max_train_samples=args.train_samples,
            max_val_samples=args.val_samples,
        )

        def on_epoch_end(epoch: int, val_loss: float) -> None:
            if not math.isfinite(val_loss):
                raise optuna.TrialPruned()
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = None
        best_val = None
        best_epoch = 0
        try:
            model = ConvNeXtV2SimMIM(
                encoder_name=args.encoder,
                pretrained=True,
                drop_path_rate=params["drop_path"],
                grad_checkpointing=args.grad_checkpointing,
            ).to(device)
            train_mask_gen = SimMIMMaskGenerator(
                IMAGE_SIZE, params["mask_patch_size"], params["mask_ratio"]
            )
            val_mask_gen = SimMIMMaskGenerator(
                IMAGE_SIZE,
                model.encoder_stride,
                args.val_mask_ratio,
                seed=args.seed,
            )
            best_val, best_epoch, _ = fit(
                model,
                train_loader,
                val_loader,
                train_mask_gen,
                val_mask_gen,
                device,
                max_epochs=args.epochs,
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                scheduler_name=params["scheduler"],
                warmup_epochs=params["warmup_epochs"],
                grad_clip=params["grad_clip"],
                patience=None,
                keep_state=False,
                on_epoch_end=on_epoch_end,
                show_progress=False,
            )
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Trial {trial.number}: CUDA out of memory -> pruned.")
            best_val = None
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if best_val is None or not math.isfinite(best_val):
            raise optuna.TrialPruned()

        trial.set_user_attr("best_epoch", best_epoch)
        return best_val

    return objective


def main(args: argparse.Namespace) -> None:
    device = get_device()
    logger.info(f"Using device: {device}")

    # Build the dataset once: scanning the disk for DICOM files is slow.
    dataset = MammoDataset(csv_file=Path(args.csv), ssl_mode=True)
    logger.info(f"Images available for SSL: {len(dataset)}")

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, multivariate=True, n_startup_trials=args.startup_trials
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.startup_trials, n_warmup_steps=3
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,  # SQLite: the study can be resumed after a crash.
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        # Include the original configuration as a baseline trial.
        study.enqueue_trial(
            {
                "lr": 1e-4,
                "weight_decay": 0.05,
                "mask_ratio": 0.6,
                "mask_patch_size": 32,
                "scheduler": "cosine",
                "warmup_epochs": 0,
                "drop_path": 0.0,
                "grad_clip": 0.0,
                "batch_size": args.batch_sizes[0],
            }
        )

    study.optimize(
        make_objective(dataset, args, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning("No trial completed; nothing to report.")
        return

    best = study.best_trial
    logger.info(f"Best trial: #{best.number} | val loss {best.value:.5f}")
    logger.info(f"Best params: {best.params}")

    # best_params.json keys are CLI argument names of c_train_mae_convnextv2.py.
    best_params = dict(best.params)
    best_params.setdefault("warmup_epochs", 0)
    with open(args.best_json, "w") as fh:
        json.dump(best_params, fh, indent=2)
    study.trials_dataframe().to_csv(args.trials_csv, index=False)
    logger.info(f"Saved {args.best_json} and {args.trials_csv}")

    try:
        importances = optuna.importance.get_param_importances(study)
        logger.info(f"Parameter importances: {importances}")
    except Exception as exc:  # Needs several completed trials.
        logger.info(f"Could not compute parameter importances: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna search for SimMIM hyperparameters.")
    parser.add_argument(
        "--csv", type=str, default="../Experimentos/Mammo_Bench_v2/mammo-bench.csv"
    )
    parser.add_argument("--encoder", type=str, default="convnextv2_base.fcmae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None, help="Seconds.")
    parser.add_argument("--startup-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per trial.")
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-mask-ratio", type=float, default=0.6)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--study-name", type=str, default="simmim_convnextv2_base")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_ssl.db")
    parser.add_argument("--best-json", type=str, default="best_params.json")
    parser.add_argument("--trials-csv", type=str, default="optuna_trials.csv")
    parser.add_argument("--log-file", type=str, default="optuna_ssl.log")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s: %(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    main(args)
