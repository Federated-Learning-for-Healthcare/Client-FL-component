"""
multimodal_loader.py — MultiModal DataLoader for federated learning.

Loads one or more modalities and returns dict-batch DataLoaders where each
batch is a dict: {"ecg": tensor, "ehr": tensor, "label": tensor}.
Only enabled modalities appear as keys — absent modality keys are omitted
and handled by the model's null-token mechanism.

Supported modality sources:
  ECG  — MIT-BIH CSV flat (187 features + label column)
         or WFDB raw waveforms (set mode: waveform, requires wfdb package)
  EHR  — CSV file (auto-detects label column)
         or UCI ML Repository (auto-downloads via ucimlrepo)
  MRI  — Image directory (JPEG/PNG, grayscale or RGB)

Multi-modality alignment:
  When more than one modality is enabled, all must provide the same number
  of samples. Samples are aligned row-by-row: row i in ECG corresponds to
  row i in EHR. If counts differ, a ValueError is raised.
  For independent single-modality clients (the common FL case), only one
  modality need be enabled.

Config example (YAML):
  data:
    type: multimodal
    params:
      batch_size: 32
      test_split: 0.2
      num_clients: 1
      client_id: 0
      random_seed: 42
      modalities:
        ecg:
          enabled: true
          mode: csv_flat        # csv_flat | waveform
          data_dir: ./data/mitbih
          n_features: 187
        mri:
          enabled: false
        ehr:
          enabled: false
          mode: ucimlrepo       # ucimlrepo | csv
          ucimlrepo_id: 45
          n_features: 13
          # csv_path: ./data/heart.csv  (used when mode: csv)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.interfaces import AbstractDataLoader

logger = logging.getLogger(__name__)


# ── Dict-batch Dataset ────────────────────────────────────────────────────────

class MultiModalDataset(Dataset):
    """
    Dataset that yields dict batches.

    Args:
        data   : dict mapping modality name → np.ndarray of shape (N, features)
        labels : np.ndarray of shape (N,)
    """

    def __init__(self, data: Dict[str, np.ndarray], labels: np.ndarray):
        self.data   = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        lengths = [len(v) for v in data.values()]
        assert all(l == lengths[0] for l in lengths), "All modalities must have same N"
        assert len(labels) == lengths[0], "Labels length must match modality data"

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        sample = {mod: tensor[idx] for mod, tensor in self.data.items()}
        sample["label"] = self.labels[idx]
        return sample


def _collate_multimodal(samples: List[dict]) -> dict:
    """Stack list-of-dicts into dict-of-batched-tensors."""
    batch: dict = {}
    for key in samples[0]:
        batch[key] = torch.stack([s[key] for s in samples])
    return batch


# ── Raw data loading helpers ──────────────────────────────────────────────────

def _load_ecg_flat_raw(
    data_dir:              str,
    n_features:            int = 187,
    balance_classes:       bool = True,
    max_samples_per_class: int  = 5000,
    random_seed:           int  = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MIT-BIH flat CSV.  Returns (X_train, y_train, X_test, y_test).
    Files expected: {data_dir}/mitbih_train.csv  and  mitbih_test.csv
    """
    import pandas as pd

    train_path = Path(data_dir) / "mitbih_train.csv"
    test_path  = Path(data_dir) / "mitbih_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"MIT-BIH CSV files not found in {data_dir}.\n"
            "Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
            "Files needed: mitbih_train.csv  and  mitbih_test.csv"
        )

    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path,  header=None)

    X_tr = train_df.iloc[:, :n_features].values.astype(np.float32)
    y_tr = train_df.iloc[:, n_features].values.astype(np.int64)
    X_te = test_df.iloc[:,  :n_features].values.astype(np.float32)
    y_te = test_df.iloc[:,  n_features].values.astype(np.int64)

    if balance_classes:
        X_tr, y_tr = _balance(X_tr, y_tr, max_samples_per_class, random_seed)

    return X_tr, y_tr, X_te, y_te


def _load_ehr_raw(
    mode:         str,
    csv_path:     Optional[str] = None,
    ucimlrepo_id: int = 45,
    random_seed:  int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load EHR data.  Returns (X, y) as float32 / int64 arrays."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    if mode == "ucimlrepo":
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise ImportError("Install ucimlrepo: pip install ucimlrepo")
        logger.info("MultiModalLoader: fetching EHR from UCI (id=%d)...", ucimlrepo_id)
        ds    = fetch_ucirepo(id=ucimlrepo_id)
        X_raw = ds.data.features.values.astype(float)
        y_raw = ds.data.targets.values.ravel()
    elif mode == "csv":
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(f"EHR CSV not found: {csv_path}")
        logger.info("MultiModalLoader: reading EHR CSV %s", csv_path)
        df = pd.read_csv(csv_path).replace("?", np.nan)
        target_col = next(
            (c for c in ["target", "num", "condition", "output"] if c in df.columns),
            df.columns[-1],
        )
        X_raw = df[[c for c in df.columns if c != target_col]].astype(float).values
        y_raw = df[target_col].astype(float).values
    else:
        raise ValueError(f"EHR mode must be 'ucimlrepo' or 'csv', got '{mode}'")

    # Impute missing, binarise, standardise
    df_x = pd.DataFrame(X_raw).apply(lambda col: col.fillna(col.median()), axis=0)
    X    = StandardScaler().fit_transform(df_x.values).astype(np.float32)
    y    = (np.array(y_raw).ravel() > 0).astype(np.int64)

    logger.info("MultiModalLoader: EHR — N=%d  features=%d  pos=%.1f%%",
                len(X), X.shape[1], 100.0 * y.mean())
    return X, y


def _load_mri_raw(
    image_dir:   str,
    img_size:    int = 64,
    in_channels: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MRI images from a directory.  Returns (X, y) where X is (N, C, H, W).

    Expected directory layout (sub-folder = class label):
      image_dir/
        0/  img001.jpg  img002.png  ...
        1/  img003.jpg  ...
    """
    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError:
        raise ImportError(
            "PIL and torchvision are required for MRI loading.\n"
            "Install with: pip install pillow torchvision"
        )

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])

    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"MRI image directory not found: {image_dir}")

    images, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_dir in sorted(image_dir_path.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            label = int(class_dir.name)
        except ValueError:
            logger.warning("MRI: skipping non-integer directory '%s'", class_dir.name)
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            mode = "L" if in_channels == 1 else "RGB"
            img  = Image.open(img_path).convert(mode)
            images.append(transform(img).numpy())
            labels.append(label)

    if not images:
        raise ValueError(
            f"No images found in {image_dir}. "
            "Expected sub-folders named by integer class labels."
        )

    X = np.stack(images, axis=0).astype(np.float32)  # (N, C, H, W)
    y = np.array(labels, dtype=np.int64)
    logger.info("MultiModalLoader: MRI — N=%d  shape=%s", len(X), X.shape)
    return X, y


def _balance(
    X: np.ndarray, y: np.ndarray,
    max_per_class: int, random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cap majority class and oversample minority classes."""
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)
    X_parts, y_parts = [], []
    for cls in classes:
        idx     = np.where(y == cls)[0]
        target  = min(max_per_class, max(len(idx), 1))
        chosen  = rng.choice(idx, size=target, replace=len(idx) < target)
        X_parts.append(X[chosen]);  y_parts.append(y[chosen])
    X_bal = np.concatenate(X_parts)
    y_bal = np.concatenate(y_parts)
    perm  = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


def _partition(
    X: np.ndarray, y: np.ndarray,
    num_clients: int, client_id: int, random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """IID shard for FL simulation."""
    rng    = np.random.default_rng(random_seed)
    shards = np.array_split(rng.permutation(len(X)), num_clients)
    if client_id >= len(shards):
        raise ValueError(
            f"client_id={client_id} out of range for num_clients={num_clients}"
        )
    idx = shards[client_id]
    return X[idx], y[idx]


# ── Main DataLoader class ─────────────────────────────────────────────────────

class MultiModalDataLoader(AbstractDataLoader):
    """
    DataLoader for one or more healthcare modalities.

    Returns dict-batch (train_loader, test_loader) where batches contain
    only the enabled modalities' tensors plus a 'label' key.

    When multiple modalities are enabled, all must have equal sample counts
    (aligned row-by-row — same patient across files).
    """

    def __init__(
        self,
        batch_size:  int   = 32,
        test_split:  float = 0.2,
        num_clients: int   = 1,
        client_id:   int   = 0,
        random_seed: int   = 42,
        modalities:  Dict[str, Any] = None,
    ):
        self.batch_size  = batch_size
        self.test_split  = test_split
        self.num_clients = num_clients
        self.client_id   = client_id
        self.random_seed = random_seed
        self.modalities  = modalities or {}

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load enabled modalities, align, partition, split, return loaders."""
        train_data: Dict[str, np.ndarray] = {}
        test_data:  Dict[str, np.ndarray] = {}
        train_labels = test_labels = None

        ecg_cfg = self.modalities.get("ecg", {})
        mri_cfg = self.modalities.get("mri", {})
        ehr_cfg = self.modalities.get("ehr", {})

        # ── ECG ──────────────────────────────────────────────────────────────
        if ecg_cfg.get("enabled", False):
            mode = ecg_cfg.get("mode", "csv_flat")
            if mode == "csv_flat":
                X_tr, y_tr, X_te, y_te = _load_ecg_flat_raw(
                    data_dir              = ecg_cfg.get("data_dir", "./data/mitbih"),
                    n_features            = ecg_cfg.get("n_features", 187),
                    balance_classes       = ecg_cfg.get("balance_classes", True),
                    max_samples_per_class = ecg_cfg.get("max_samples_per_class", 5000),
                    random_seed           = self.random_seed,
                )
                if self.num_clients > 1:
                    X_tr, y_tr = _partition(X_tr, y_tr, self.num_clients,
                                            self.client_id, self.random_seed)
                train_data["ecg"] = X_tr;  train_labels = y_tr
                test_data["ecg"]  = X_te;  test_labels  = y_te
            elif mode == "waveform":
                raise NotImplementedError(
                    "ECG waveform mode requires wfdb and WFDB-format records. "
                    "Use mode: csv_flat for MIT-BIH CSV data."
                )
            else:
                raise ValueError(f"Unknown ECG mode '{mode}'. Use 'csv_flat' or 'waveform'.")

        # ── EHR ──────────────────────────────────────────────────────────────
        if ehr_cfg.get("enabled", False):
            X, y = _load_ehr_raw(
                mode         = ehr_cfg.get("mode", "ucimlrepo"),
                csv_path     = ehr_cfg.get("csv_path"),
                ucimlrepo_id = ehr_cfg.get("ucimlrepo_id", 45),
                random_seed  = self.random_seed,
            )
            if self.num_clients > 1:
                X, y = _partition(X, y, self.num_clients, self.client_id, self.random_seed)

            # Split into train/test
            n_te    = max(1, int(len(X) * self.test_split))
            n_tr    = len(X) - n_te
            rng     = np.random.default_rng(self.random_seed)
            perm    = rng.permutation(len(X))
            tr_idx  = perm[:n_tr];  te_idx = perm[n_tr:]
            train_data["ehr"] = X[tr_idx];  test_data["ehr"] = X[te_idx]
            if train_labels is None:
                train_labels = y[tr_idx];  test_labels = y[te_idx]

        # ── MRI ──────────────────────────────────────────────────────────────
        if mri_cfg.get("enabled", False):
            X, y = _load_mri_raw(
                image_dir   = mri_cfg.get("image_dir", "./data/mri"),
                img_size    = mri_cfg.get("img_size", 64),
                in_channels = mri_cfg.get("in_channels", 1),
            )
            if self.num_clients > 1:
                X, y = _partition(X, y, self.num_clients, self.client_id, self.random_seed)

            n_te   = max(1, int(len(X) * self.test_split))
            n_tr   = len(X) - n_te
            rng    = np.random.default_rng(self.random_seed)
            perm   = rng.permutation(len(X))
            tr_idx = perm[:n_tr];  te_idx = perm[n_tr:]
            train_data["mri"] = X[tr_idx];  test_data["mri"] = X[te_idx]
            if train_labels is None:
                train_labels = y[tr_idx];  test_labels = y[te_idx]

        if not train_data:
            raise ValueError(
                "No modalities enabled. Set at least one modality's 'enabled: true' "
                "in the data.params.modalities config block."
            )

        # Verify all enabled modalities have the same sample count
        self._check_alignment(train_data, "train")
        self._check_alignment(test_data,  "test")

        train_ds = MultiModalDataset(train_data, train_labels)
        test_ds  = MultiModalDataset(test_data,  test_labels)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, drop_last=False, collate_fn=_collate_multimodal,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size,
            shuffle=False, drop_last=False, collate_fn=_collate_multimodal,
        )

        logger.info(
            "MultiModalDataLoader: train=%d  test=%d  modalities=%s  batch_size=%d",
            len(train_ds), len(test_ds), list(train_data.keys()), self.batch_size,
        )
        return train_loader, test_loader

    @staticmethod
    def _check_alignment(data: Dict[str, np.ndarray], split: str) -> None:
        counts = {k: len(v) for k, v in data.items()}
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"Multimodal {split} data misaligned — sample counts differ: {counts}. "
                "All enabled modalities must have the same number of samples "
                "(aligned row-by-row). Check your data files."
            )
