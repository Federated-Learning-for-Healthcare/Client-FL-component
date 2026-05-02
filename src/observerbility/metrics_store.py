"""
metrics_store.py — Per-federation CSV metrics recorder for paper experiments.

Writes one CSV file per federation with one row per training round.
Designed to be directly importable into pandas for paper plots.

Columns recorded:
  federation_id, global_round, client_round, timestamp,
  train_loss, train_accuracy, eval_loss, eval_accuracy,
  update_size_bytes, compression_ratio,
  privacy_type, noise_multiplier, clipping_norm,
  model_type, param_count,
  elapsed_seconds, ram_usage_mb, cpu_percent

Usage:
  metrics = MetricsStore(federation_id="heart_disease", output_dir="output/metrics")
  metrics.record_round(global_round=1, train_loss=0.45, ...)

  # In Python / Jupyter for paper plots:
  import pandas as pd
  df = pd.read_csv("output/metrics/heart_disease_metrics.csv")
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# All columns in order — do not reorder, CSV format depends on this
_COLUMNS = [
    "federation_id",
    "global_round",
    "client_round",
    "timestamp",
    "train_loss",
    "train_accuracy",
    "eval_loss",
    "eval_accuracy",
    "update_size_bytes",
    "compression_ratio",
    "privacy_type",
    "noise_multiplier",
    "clipping_norm",
    "model_type",
    "param_count",
    "elapsed_seconds",
    "ram_usage_mb",
    "cpu_percent",
]


@dataclass
class RoundMetrics:
    """One row of metrics for one training round."""
    federation_id:     str
    global_round:      Optional[int]   = None
    client_round:      Optional[int]   = None
    timestamp:         Optional[str]   = None
    train_loss:        Optional[float] = None
    train_accuracy:    Optional[float] = None
    eval_loss:         Optional[float] = None
    eval_accuracy:     Optional[float] = None
    update_size_bytes: Optional[int]   = None
    compression_ratio: Optional[float] = None
    privacy_type:      Optional[str]   = None
    noise_multiplier:  Optional[float] = None
    clipping_norm:     Optional[float] = None
    model_type:        Optional[str]   = None
    param_count:       Optional[int]   = None
    elapsed_seconds:   Optional[float] = None
    ram_usage_mb:      Optional[float] = None
    cpu_percent:       Optional[float] = None


class MetricsStore:
    """
    Thread-safe CSV metrics recorder for one federation.

    Creates output_dir/federation_id_metrics.csv on first write.
    Appends one row per round — safe to read while training is running.

    Parameters
    ----------
    federation_id : str
        Unique identifier for this federation (used in filename + column)
    output_dir    : str
        Directory to write CSV files into
    privacy_type      : str   — recorded in every row
    noise_multiplier  : float — recorded in every row
    clipping_norm     : float — recorded in every row
    model_type        : str   — recorded in every row
    param_count       : int   — recorded in every row
    """

    def __init__(
        self,
        federation_id:    str,
        output_dir:       str   = "output/metrics",
        privacy_type:     str   = "none",
        noise_multiplier: float = 0.0,
        clipping_norm:    float = 0.0,
        model_type:       str   = "unknown",
        param_count:      int   = 0,
    ):
        self.federation_id    = federation_id
        self.output_dir       = Path(output_dir)
        self.privacy_type     = privacy_type
        self.noise_multiplier = noise_multiplier
        self.clipping_norm    = clipping_norm
        self.model_type       = model_type
        self.param_count      = param_count
        self._lock            = threading.Lock()
        self._round_start_time: Optional[float] = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / f"{federation_id}_metrics.csv"

        # Write header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                writer.writeheader()
            logger.info(
                "MetricsStore [%s]: created %s", federation_id, self.csv_path
            )
        else:
            logger.info(
                "MetricsStore [%s]: appending to %s", federation_id, self.csv_path
            )

    def mark_round_start(self) -> None:
        """Call at the start of each round to enable elapsed_seconds tracking."""
        self._round_start_time = time.time()

    def record_round(
        self,
        global_round:      Optional[int]   = None,
        client_round:      Optional[int]   = None,
        train_loss:        Optional[float] = None,
        train_accuracy:    Optional[float] = None,
        eval_loss:         Optional[float] = None,
        eval_accuracy:     Optional[float] = None,
        update_size_bytes: Optional[int]   = None,
        compression_ratio: Optional[float] = None,
    ) -> None:
        """
        Record metrics for one completed round. Appends one row to CSV.
        Thread-safe — can be called from the worker thread.
        """
        elapsed = (
            round(time.time() - self._round_start_time, 3)
            if self._round_start_time is not None
            else None
        )
        ram_mb, cpu_pct = self._system_stats()

        row = RoundMetrics(
            federation_id     = self.federation_id,
            global_round      = global_round,
            client_round      = client_round,
            timestamp         = time.strftime("%Y-%m-%dT%H:%M:%S"),
            train_loss        = _r(train_loss),
            train_accuracy    = _r(train_accuracy),
            eval_loss         = _r(eval_loss),
            eval_accuracy     = _r(eval_accuracy),
            update_size_bytes = update_size_bytes,
            compression_ratio = _r(compression_ratio),
            privacy_type      = self.privacy_type,
            noise_multiplier  = self.noise_multiplier,
            clipping_norm     = self.clipping_norm,
            model_type        = self.model_type,
            param_count       = self.param_count,
            elapsed_seconds   = elapsed,
            ram_usage_mb      = ram_mb,
            cpu_percent       = cpu_pct,
        )

        with self._lock:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS)
                writer.writerow(asdict(row))

        logger.debug(
            "MetricsStore [%s]: round %s recorded — acc=%.4f  loss=%.4f",
            self.federation_id,
            global_round,
            train_accuracy or 0.0,
            train_loss or 0.0,
        )

    def read_all(self) -> list:
        """
        Read all recorded rounds as a list of dicts.
        Useful for the status API — returns same data the CSV contains.
        """
        with self._lock:
            if not self.csv_path.exists():
                return []
            with open(self.csv_path, "r", newline="") as f:
                return list(csv.DictReader(f))

    def csv_file_path(self) -> str:
        """Return the absolute path to the CSV file."""
        return str(self.csv_path.resolve())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _system_stats(self):
        """Capture RAM and CPU usage. Returns (ram_mb, cpu_pct) or (None, None)."""
        try:
            import psutil
            proc    = psutil.Process(os.getpid())
            ram_mb  = round(proc.memory_info().rss / 1024 / 1024, 2)
            cpu_pct = round(psutil.cpu_percent(interval=None), 1)
            return ram_mb, cpu_pct
        except ImportError:
            return None, None


def _r(v: Optional[float], decimals: int = 6) -> Optional[float]:
    """Round a float for CSV storage. Pass-through for None."""
    return round(v, decimals) if v is not None else None


def compute_update_size_bytes(weights) -> int:
    """
    Compute the total byte size of a list of numpy weight arrays.
    Call this in client.py after compression to record actual transmission size.
    """
    import numpy as np
    return int(sum(w.nbytes for w in weights if hasattr(w, "nbytes")))