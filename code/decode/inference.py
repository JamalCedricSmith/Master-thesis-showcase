"""Inference utilities: load a trained model and return raw logits."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.extract import load_dataset
from data.transform import transform_dataset
from data.split import split_dataset, split_dvss_unseen_protocol
from data.dataloader import make_dataloader, Batch
from features.preprocess import preprocess_batch
from features.features import extract_features_batch


def _get_builder(model_name: str):
    # Lazy import to avoid stale modules in notebooks.
    module = importlib.import_module(f"models.{model_name}")
    if not hasattr(module, "build_model"):
        module = importlib.reload(module)
    if not hasattr(module, "build_model"):
        raise ImportError(f"models.{model_name} does not define build_model(...)")
    return getattr(module, "build_model")


def _seed_worker(worker_id: int) -> None:
    # Deterministic worker init for reproducible batching.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=device)
    if "config" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint missing config or model_state.")
    return ckpt


def _select_split(model_df, *, split: str, train_cfg: dict[str, Any], dataset_cfg: dict[str, Any]):
    if train_cfg.get("use_banlist", False):
        repo_root = Path(__file__).resolve().parent.parent
        banlist_path = repo_root / dataset_cfg.get("banlist_path", "data/banlist.txt")
        train_df, val_df, test_df = split_dvss_unseen_protocol(
            model_df,
            banlist_path=banlist_path,
            seed=train_cfg.get("split_seed", train_cfg["seed"]),
            train_ratio=float(train_cfg.get("train_ratio", 0.8)),
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.1)),
            shuffle=train_cfg.get("split_shuffle", True),
        )
    else:
        train_df, val_df, test_df = split_dataset(
            model_df,
            seed=train_cfg.get("split_seed", train_cfg["seed"]),
            train_ratio=float(train_cfg.get("train_ratio", 0.8)),
            val_ratio=float(train_cfg.get("val_ratio", 0.1)),
            test_ratio=float(train_cfg.get("test_ratio", 0.1)),
            shuffle=train_cfg.get("split_shuffle", True),
        )
    
    if split == "train":
        return train_df
    if split == "val":
        return val_df
    if split == "test":
        return test_df
    raise ValueError("split must be one of: train, val, test")


def predict_logits(
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    batch_size: int | None = None,
    max_batches: int | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Run inference and return raw logits with metadata.

    Returns a dict with:
    - results: list of batch dicts (logits, out_len, labels, label_len, trial_id)
    - config: merged config used during training
    - vocab: list of token symbols (no blank)
    - model_df: DataFrame used to build splits (for text lookup)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_checkpoint(checkpoint_path, device=device)
    cfg = ckpt["config"]
    vocab = ckpt.get("vocab", [])
    
    # Load global normalization stats from the same directory as the checkpoint
    ckpt_dir = Path(checkpoint_path).parent
    mean_path = ckpt_dir / "global_mean.npy"
    std_path = ckpt_dir / "global_std.npy"
    global_stats_np = None
    if mean_path.exists() and std_path.exists():
        g_mean = np.load(mean_path)
        g_std = np.load(std_path)
        global_stats_np = (g_mean, g_std)
        print(f"Loaded global normalization stats from {ckpt_dir}")
    else:
        print("Warning: Global normalization stats not found. Falling back to per-utterance.")

    # IMPORTANT: always use the dataset settings from the checkpoint config
    # to avoid mixing CSL/DVSS between training and inference.
    ds_cfg = {k: v for k, v in cfg["dataset"].items() if v is not None}
    dataset_name = ds_cfg.pop("name")
    df = load_dataset(dataset_name, **ds_cfg)
    train_cfg = cfg["training"]

    # Match training data construction exactly: subsample before transform/split.
    if train_cfg.get("sample_n") is not None:
        df = df.head(train_cfg["sample_n"])

    transform_cfg = cfg["transform"]
    ignore_last = transform_cfg.get("ignore_last_emg_channel", False)
    if ignore_last == "auto":
        ignore_last = dataset_name.lower() == "csl"

    model_df, _, _ = transform_dataset(
        df,
        token_id_offset=transform_cfg.get("token_id_offset", 1),
        ignore_last_emg_channel=ignore_last,
        drop_empty_tokens=transform_cfg.get("drop_empty_tokens", True),
        # [CENSORED / PROPRIETARY] Target formulation parameter withheld
    )
    # Ensure deterministic order before splitting.
    if not model_df.empty:
        model_df = model_df.sort_values("trial_id").reset_index(drop=True)

    feat_cfg = cfg.get("features", {})
    data_df = _select_split(model_df, split=split, train_cfg=train_cfg, dataset_cfg=cfg["dataset"])

    pre_cfg = cfg.get("preprocess", {})

    g = torch.Generator()
    g.manual_seed(train_cfg["seed"])
    dl = make_dataloader(
        data_df,
        batch_size=batch_size or train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        generator=g,
        worker_init_fn=_seed_worker,
        normalize_method=pre_cfg.get("normalize_method") if pre_cfg.get("normalize") else None,
        stats=global_stats_np
    )

    model_cfg = cfg["model"]
    builder = _get_builder(model_cfg["name"])
    model = None

    results: list[dict[str, Any]] = []
    
    for step, batch in enumerate(dl):
        if max_batches is not None and step >= max_batches:
            break

        # Keep a copy of raw EMG if it was loaded
        raw_emg = batch.emg.clone()
        raw_emg_len = batch.emg_len.clone()

        # DataLoader already normalized batch.emg. 
        # Preprocess handles batch formatting.
        batch = preprocess_batch(
            batch,
            train=False,
            normalize=False, # Handled by DataLoader
        )
        # Use same feature extraction as training.
        batch = extract_features_batch(batch, **feat_cfg)

        if model is None:
            model = builder(
                feature_dim=batch.emg.shape[-1],
                # Match train.py model initialization exactly.
                vocab_size=len(vocab) + 1,
                cfg=model_cfg,
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

        with torch.no_grad():
            logits, out_len = model(batch.emg.to(device), batch.emg_len.to(device))

        results.append(
            {
                "logits": logits.cpu(),
                "out_len": out_len.cpu(),
                "labels": batch.labels_padded.cpu(),
                "label_len": batch.label_len.cpu(),
                "trial_id": batch.trial_id,
                "raw_emg": raw_emg.cpu(),
                "raw_emg_len": raw_emg_len.cpu(),
            }
        )

    return {
        "results": results,
        "config": cfg,
        "vocab": vocab,
        "model_df": model_df,
        "split": split,
        "split_size": int(len(data_df)),
    }


def save_logits_json(out: dict[str, Any], path: str | Path) -> None:
    """Optional helper to persist logits to disk in a simple JSON format."""
    payload = {
        "results": [
            {
                "logits": r["logits"].numpy(),
                "out_len": r["out_len"].numpy(),
                "labels": r["labels"].numpy(),
                "label_len": r["label_len"].numpy(),
                "trial_id": r["trial_id"],
            }
            for r in out["results"]
        ],
        "config": out["config"],
        "vocab": out["vocab"],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=True, indent=2))
