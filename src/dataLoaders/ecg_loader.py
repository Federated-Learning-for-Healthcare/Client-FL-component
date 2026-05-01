"""
ecg_loader.py — ECG DataLoader for time-series arrhythmia datasets.

Supports the Kaggle ECG Heartbeat Categorization dataset (MIT-BIH subset):
  87,554 training + 21,892 test samples
  188 columns: 187 time-step signal values + 1 class label
  5-class arrhythmia classification:
    0 = Normal
    1 = Supraventricular premature beat
    2 = Premature ventricular contraction
    3 = Fusion of ventricular and normal beat
    4 = Unclassifiable beat
  Severely imbalanced: ~83% class 0

Download:
  https://www.kaggle.com/datasets/shayanfazeli/heartbeat
  Files needed: mitbih_train.csv and mitbih_test.csv
  Place in data_dir (default: ./data/mitbih/)

Federated simulation:
  Set num_clients and client_id to partition training data across
  simulated hospitals. Test set is always the full held-out set.

Config example:
  data:
    type: ecg
    params:
      data_dir: ./data/mitbih
      batch_size: 64
      balance_classes: true
      max_samples_per_class: 5000
      num_clients: 1
      client_id: 0
      random_seed: 42
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.interfaces import AbstractDataLoader

logger = logging.getLogger(__name__)

NUM_FEATURES = 187
NUM_CLASSES  = 5

CLASS_NAMES = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unclassifiable",
}


class ECGLoader(AbstractDataLoader):
    """
    DataLoader for the MIT-BIH Arrhythmia ECG dataset (Kaggle CSV version).

    Input dimension : 187 (one ECG beat time series, normalised 0–1)
    Output classes  : 5 (arrhythmia types)
    """

    def __init__(
        self,
        data_dir:               str   = "./data/mitbih",
        batch_size:             int   = 64,
        balance_classes:        bool  = True,
        max_samples_per_class:  int   = 5000,
        num_clients:            int   = 1,
        client_id:              int   = 0,
        random_seed:            int   = 42,
        shuffle:                bool  = True,
        refresh_each_round:     bool  = False,
    ):
        self.data_dir              = Path(data_dir)
        self.batch_size            = batch_size
        self.balance_classes       = balance_classes
        self.max_samples_per_class = max_samples_per_class
        self.num_clients           = num_clients
        self.client_id             = client_id
        self.random_seed           = random_seed
        self.shuffle               = shuffle
        self.refresh_each_round    = refresh_each_round

        # Cache for refresh_each_round mode
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_test:  Optional[np.ndarray] = None
        self._y_test:  Optional[np.ndarray] = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and return (train_loader, test_loader)."""

        if self._X_train is None or self.refresh_each_round:
            self._load_raw()

        X_train, y_train = self._X_train, self._y_train
        X_test,  y_test  = self._X_test,  self._y_test

        # Class balancing on training set only
        if self.balance_classes:
            X_train, y_train = self._balance(X_train, y_train)
            self._log_distribution("train (balanced)", y_train)

        # FL partition on training set only
        # Test set is always the full held-out set for consistent evaluation
        if self.num_clients > 1:
            X_train, y_train = self._partition(X_train, y_train)
            logger.info(
                "ECGLoader: client %d/%d — %d training samples after partition",
                self.client_id, self.num_clients, len(X_train),
            )

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_ds = TensorDataset(
            torch.tensor(X_test,  dtype=torch.float32),
            torch.tensor(y_test,  dtype=torch.long),
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=self.shuffle, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size,
            shuffle=False, drop_last=False,
        )

        logger.info(
            "ECGLoader: train=%d  test=%d  features=%d  batch_size=%d",
            len(train_ds), len(test_ds), NUM_FEATURES, self.batch_size,
        )
        return train_loader, test_loader

    def feature_dim(self) -> int:
        return NUM_FEATURES

    def num_classes(self) -> int:
        return NUM_CLASSES

    # ------------------------------------------------------------------
    # Raw loading
    # ------------------------------------------------------------------

    def _load_raw(self) -> None:
        import pandas as pd

        train_path = self.data_dir / "mitbih_train.csv"
        test_path  = self.data_dir / "mitbih_test.csv"

        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"MIT-BIH CSV files not found in {self.data_dir}\n"
                "Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat\n"
                "Files needed: mitbih_train.csv and mitbih_test.csv"
            )

        logger.info("ECGLoader: reading %s ...", train_path)
        train_df = pd.read_csv(train_path, header=None)
        test_df  = pd.read_csv(test_path,  header=None)

        self._X_train = train_df.iloc[:, :NUM_FEATURES].values.astype(np.float32)
        self._y_train = train_df.iloc[:, NUM_FEATURES].values.astype(np.int64)
        self._X_test  = test_df.iloc[:,  :NUM_FEATURES].values.astype(np.float32)
        self._y_test  = test_df.iloc[:,  NUM_FEATURES].values.astype(np.int64)

        self._log_distribution("train (raw)", self._y_train)
        self._log_distribution("test",        self._y_test)

    # ------------------------------------------------------------------
    # Class balancing
    # ------------------------------------------------------------------

    def _balance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance class distribution:
          - Cap majority class at max_samples_per_class
          - Oversample minority classes up to max_samples_per_class
        """
        rng = np.random.default_rng(self.random_seed)
        X_parts, y_parts = [], []

        for cls in range(NUM_CLASSES):
            idx = np.where(y == cls)[0]
            if len(idx) == 0:
                logger.warning("ECGLoader: class %d has no samples — skipping", cls)
                continue

            target_n = min(self.max_samples_per_class, max(len(idx), 1))
            replace  = len(idx) < target_n
            chosen   = rng.choice(idx, size=target_n, replace=replace)

            X_parts.append(X[chosen])
            y_parts.append(y[chosen])

        X_bal = np.concatenate(X_parts, axis=0)
        y_bal = np.concatenate(y_parts, axis=0)
        perm  = rng.permutation(len(X_bal))
        return X_bal[perm], y_bal[perm]

    # ------------------------------------------------------------------
    # FL partitioning
    # ------------------------------------------------------------------

    def _partition(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IID partition across num_clients."""
        rng     = np.random.default_rng(self.random_seed)
        indices = rng.permutation(len(X))
        shards  = np.array_split(indices, self.num_clients)

        if self.client_id >= len(shards):
            raise ValueError(
                f"client_id={self.client_id} out of range "
                f"for num_clients={self.num_clients}"
            )
        idx = shards[self.client_id]
        return X[idx], y[idx]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_distribution(self, label: str, y: np.ndarray) -> None:
        total = len(y)
        dist  = {
            f"{c}:{CLASS_NAMES[c]}": int((y == c).sum())
            for c in range(NUM_CLASSES)
            if (y == c).sum() > 0
        }
        logger.info("ECGLoader [%s]: total=%d  %s", label, total, dist)