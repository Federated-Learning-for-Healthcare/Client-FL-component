"""
ehr_loader.py — EHR DataLoader for tabular clinical datasets.

Supports any tabular heart disease dataset with the standard 13-feature
Cleveland format. Works with:
  - UCI Cleveland Heart Disease  (303 rows, 13 features)
  - UCI Hungarian Heart Disease  (294 rows, 13 features)
  - South Africa Heart Disease   (462 rows, 9 features — auto-detected)
  - Heart Failure Clinical Records (299 rows, 12 features)
  - Any compatible CSV with a binary or multi-class target column

Download options:
  Option A — ucimlrepo (auto, recommended):
    pip install ucimlrepo scikit-learn
    set data_source: ucimlrepo

  Option B — local CSV:
    set data_source: csv
    set data_path: ./data/heart.csv

Federated simulation:
  Set num_clients and client_id to partition data across simulated hospitals.
  E.g. num_clients=3, client_id=0 gives hospital 0 its shard.

Config example:
  data:
    type: ehr
    params:
      data_source: ucimlrepo      # or: csv
      data_path: ./data/heart.csv # used when data_source: csv
      batch_size: 32
      test_split: 0.2
      num_clients: 1              # set > 1 for FL simulation
      client_id: 0                # which shard this hospital gets
      random_seed: 42
      refresh_each_round: false   # set true for dynamic data loading
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.core.interfaces import AbstractDataLoader

logger = logging.getLogger(__name__)


class EHRLoader(AbstractDataLoader):
    """
    DataLoader for tabular EHR heart disease datasets.

    Returns standardised float32 tensors with binary labels (0/1).
    Handles missing values, standardisation, and FL data partitioning.

    Input dimension: determined by dataset (13 for Cleveland, etc.)
    Output classes:  2 (no disease / disease)
    """

    def __init__(
        self,
        data_source:        str   = "ucimlrepo",
        data_path:          str   = "./data/heart.csv",
        ucimlrepo_id:       int   = 45,
        batch_size:         int   = 32,
        test_split:         float = 0.2,
        num_clients:        int   = 1,
        client_id:          int   = 0,
        random_seed:        int   = 42,
        shuffle:            bool  = True,
        refresh_each_round: bool  = False,
    ):
        self.data_source        = data_source
        self.data_path          = Path(data_path)
        self.ucimlrepo_id       = ucimlrepo_id
        self.batch_size         = batch_size
        self.test_split         = test_split
        self.num_clients        = num_clients
        self.client_id          = client_id
        self.random_seed        = random_seed
        self.shuffle            = shuffle
        self.refresh_each_round = refresh_each_round

        # Cache loaded data for refresh_each_round mode
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load, preprocess, partition, and return (train_loader, test_loader).

        If refresh_each_round=True, re-queries data source each call.
        Otherwise uses cached data after first load.
        """
        if self._X is None or self.refresh_each_round:
            self._X, self._y = self._load_raw()

        X, y = self._X, self._y

        # Partition for FL simulation
        if self.num_clients > 1:
            X, y = self._partition(X, y)
            logger.info(
                "EHRLoader: client %d/%d — %d samples after partition",
                self.client_id, self.num_clients, len(X),
            )

        # Build dataset and split
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        n_test  = max(1, int(len(dataset) * self.test_split))
        n_train = len(dataset) - n_test

        gen = torch.Generator().manual_seed(self.random_seed)
        train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=self.shuffle, drop_last=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size,
            shuffle=False, drop_last=False,
        )

        logger.info(
            "EHRLoader: train=%d  test=%d  features=%d  "
            "positive_class=%.1f%%  batch_size=%d",
            n_train, n_test, X.shape[1],
            100.0 * y.mean(), self.batch_size,
        )
        return train_loader, test_loader

    def feature_dim(self) -> int:
        """Return input feature dimension. Loads data if not yet loaded."""
        if self._X is None:
            self._X, self._y = self._load_raw()
        return self._X.shape[1]

    # ------------------------------------------------------------------
    # Raw loading
    # ------------------------------------------------------------------

    def _load_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw data, handle missing values, standardise, binarise target."""
        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        if self.data_source == "ucimlrepo":
            X_raw, y_raw = self._load_ucimlrepo()
        elif self.data_source == "csv":
            X_raw, y_raw = self._load_csv()
        else:
            raise ValueError(
                f"Unknown data_source '{self.data_source}'. "
                "Choose 'ucimlrepo' or 'csv'."
            )

        # Handle missing values — impute with column median
        df = pd.DataFrame(X_raw).apply(
            lambda col: col.fillna(col.median()), axis=0
        )
        X = df.values.astype(np.float32)

        # Binarise: 0 = no disease, 1 = disease (any severity)
        y = (np.array(y_raw).ravel() > 0).astype(np.int64)

        # Standardise features
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

        logger.info(
            "EHRLoader: loaded %d samples  features=%d  positive=%.1f%%",
            len(X), X.shape[1], 100.0 * y.mean(),
        )
        return X, y

    def _load_ucimlrepo(self):
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise ImportError(
                "ucimlrepo is required for data_source='ucimlrepo'.\n"
                "Install with: pip install ucimlrepo"
            )
        logger.info("EHRLoader: fetching from UCI ML Repository (id=%d)...", self.ucimlrepo_id)
        ds = fetch_ucirepo(id=self.ucimlrepo_id)
        return ds.data.features.values, ds.data.targets.values.ravel()

    def _load_csv(self):
        import pandas as pd

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.data_path}\n"
                "Download Cleveland Heart Disease from:\n"
                "  https://archive.ics.uci.edu/dataset/45/heart+disease\n"
                "Or set data_source: ucimlrepo for auto-download."
            )
        logger.info("EHRLoader: reading %s", self.data_path)

        df = pd.read_csv(self.data_path)
        df = df.replace("?", np.nan)

        # Auto-detect target column
        target_col = None
        for candidate in ["target", "num", "condition", "output"]:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            # Assume last column is target
            target_col = df.columns[-1]
            logger.warning(
                "EHRLoader: could not detect target column, using last: '%s'",
                target_col,
            )

        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].astype(float).values
        y = df[target_col].astype(float).values
        return X, y

    # ------------------------------------------------------------------
    # FL data partitioning
    # ------------------------------------------------------------------

    def _partition(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        IID partition — splits data evenly across num_clients.
        Each client gets a contiguous, non-overlapping shard.

        For non-IID partitioning (future work), sort by label before split.
        """
        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(len(X))
        shards = np.array_split(indices, self.num_clients)

        if self.client_id >= len(shards):
            raise ValueError(
                f"client_id={self.client_id} out of range "
                f"for num_clients={self.num_clients}"
            )
        shard_idx = shards[self.client_id]
        return X[shard_idx], y[shard_idx]