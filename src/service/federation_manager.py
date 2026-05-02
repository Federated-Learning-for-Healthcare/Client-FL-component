"""
federation_manager.py — Manages N concurrent FL federations on one node.

Each federation is completely isolated:
  - Separate FlowerWorker process
  - Separate StatusStore  (output/fed_{id}/status.json)
  - Separate MetricsStore (output/metrics/{id}_metrics.csv)
  - Separate CheckpointStore (output/fed_{id}/checkpoints/)
  - Separate InferenceEngine (loads latest checkpoint for prediction)
  - Separate privacy ε budget

Auto-discovery: reads all YAML files from conf/federations/ on startup.
Each file stem becomes the federation_id.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.loader import load_config
from src.observerbility.checkpoint_store import CheckpointStore
from src.observerbility.metrics_store import MetricsStore
from src.observerbility.status_store import StatusStore
from src.service.inference_engine import InferenceEngine
from src.service.worker import FlowerWorker

logger = logging.getLogger(__name__)


@dataclass
class FederationEntry:
    """Runtime state for one federation."""
    federation_id:    str
    config_path:      Path
    worker:           FlowerWorker
    status_store:     StatusStore
    metrics_store:    MetricsStore
    checkpoint_store: CheckpointStore
    inference_engine: InferenceEngine
    output_dir:       Path


class FederationManager:
    """
    Manages N concurrent FL federations on a single hospital node.

    Parameters
    ----------
    federations_dir : str — path to directory containing per-federation YAMLs
    output_base_dir : str — base directory for all output files
    """

    def __init__(
        self,
        federations_dir: str = "conf/federations",
        output_base_dir: str = "output",
    ):
        self.federations_dir = Path(federations_dir)
        self.output_base_dir = Path(output_base_dir)
        self._federations:   Dict[str, FederationEntry] = {}
        self._lock           = threading.Lock()

        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        if not self.federations_dir.exists():
            logger.warning(
                "FederationManager: federations_dir '%s' does not exist — "
                "creating with example configs.", self.federations_dir,
            )
            self.federations_dir.mkdir(parents=True, exist_ok=True)
            self._write_example_configs()
            return

        yaml_files = sorted(self.federations_dir.glob("*.yaml"))
        if not yaml_files:
            logger.warning(
                "FederationManager: no YAML files found in '%s'.",
                self.federations_dir,
            )
            return

        for config_path in yaml_files:
            self._register(config_path.stem, config_path)

        logger.info(
            "FederationManager: discovered %d federation(s): %s",
            len(self._federations), list(self._federations.keys()),
        )

    def _register(self, federation_id: str, config_path: Path) -> None:
        output_dir = self.output_base_dir / f"fed_{federation_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        status_store = StatusStore(path=str(output_dir / "status.json"))

        # Read config for MetricsStore metadata
        try:
            cfg              = load_config(config_path)
            privacy_type     = cfg.get("privacy",  {}).get("type",   "none")
            privacy_params   = cfg.get("privacy",  {}).get("params", {}) or {}
            model_type       = cfg.get("model",    {}).get("type",   "unknown")
            device           = cfg.get("runtime",  {}).get("device", "cpu")
        except Exception as e:
            logger.warning(
                "FederationManager: could not read config for '%s': %s",
                federation_id, e,
            )
            privacy_type, privacy_params, model_type, device = "none", {}, "unknown", "cpu"

        metrics_store = MetricsStore(
            federation_id    = federation_id,
            output_dir       = str(self.output_base_dir / "metrics"),
            privacy_type     = privacy_type,
            noise_multiplier = float(privacy_params.get("noise_multiplier", 0.0)),
            clipping_norm    = float(privacy_params.get("clipping_norm",    0.0)),
            model_type       = model_type,
        )

        checkpoint_store = CheckpointStore(
            federation_id  = federation_id,
            checkpoint_dir = str(output_dir / "checkpoints"),
            keep_last      = 5,
        )

        inference_engine = InferenceEngine(
            federation_id    = federation_id,
            config_path      = config_path,
            checkpoint_store = checkpoint_store,
            device           = device,
        )

        worker = FlowerWorker(
            config_path      = config_path,
            status_store     = status_store,
            metrics_store    = metrics_store,
            checkpoint_store = checkpoint_store,
            federation_id    = federation_id,
        )

        entry = FederationEntry(
            federation_id    = federation_id,
            config_path      = config_path,
            worker           = worker,
            status_store     = status_store,
            metrics_store    = metrics_store,
            checkpoint_store = checkpoint_store,
            inference_engine = inference_engine,
            output_dir       = output_dir,
        )

        with self._lock:
            self._federations[federation_id] = entry

        logger.info(
            "FederationManager: registered '%s' — config=%s",
            federation_id, config_path,
        )

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start(self, federation_id: str) -> Dict:
        entry = self._get(federation_id)
        if entry.worker.running:
            return {"started": False, "message": f"'{federation_id}' already running"}
        entry.worker.start()
        return {"started": True, "message": f"'{federation_id}' started"}

    def stop(self, federation_id: str) -> Dict:
        entry = self._get(federation_id)
        if not entry.worker.running:
            return {"stopped": False, "message": f"'{federation_id}' not running"}
        entry.worker.stop()
        return {"stopped": True, "message": f"Stop signal sent to '{federation_id}'"}

    def start_all(self) -> Dict:
        return {fid: self.start(fid) for fid in self._federations}

    def stop_all(self) -> Dict:
        return {fid: self.stop(fid) for fid in self._federations}

    def reload(self, federation_id: str) -> Dict:
        entry = self._get(federation_id)
        if entry.worker.running:
            return {"reloaded": False, "message": "Cannot reload while running"}
        self._register(federation_id, entry.config_path)
        return {"reloaded": True, "message": f"'{federation_id}' reloaded"}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, federation_id: str, inputs: list) -> Dict:
        """Run inference on the latest checkpoint for this federation."""
        entry = self._get(federation_id)
        return entry.inference_engine.predict(inputs)

    def inference_status(self, federation_id: str) -> Dict:
        """Return inference readiness info for this federation."""
        entry = self._get(federation_id)
        return {
            "federation_id": federation_id,
            "ready":         entry.inference_engine.is_ready(),
            "latest_round":  entry.inference_engine.latest_round(),
            "checkpoints":   entry.checkpoint_store.list_checkpoints(),
        }

    # ------------------------------------------------------------------
    # Status and config
    # ------------------------------------------------------------------

    def status(self, federation_id: str) -> Dict:
        entry = self._get(federation_id)
        data  = entry.status_store.read()
        data["federation_id"]    = federation_id
        data["worker_running"]   = entry.worker.running
        data["config_path"]      = str(entry.config_path)
        data["metrics_csv"]      = entry.metrics_store.csv_file_path()
        data["latest_checkpoint"] = entry.checkpoint_store.latest_round()
        data["inference_ready"]  = entry.inference_engine.is_ready()
        return data

    def status_all(self) -> Dict:
        return {
            fid: {
                "state":          self._federations[fid].status_store.read().get("state", "UNKNOWN"),
                "worker_running": self._federations[fid].worker.running,
                "config_path":    str(self._federations[fid].config_path),
                "inference_ready": self._federations[fid].inference_engine.is_ready(),
                "latest_round":   self._federations[fid].checkpoint_store.latest_round(),
            }
            for fid in self._federations
        }

    def list_federations(self) -> List[str]:
        return list(self._federations.keys())

    def get_config(self, federation_id: str) -> Dict:
        entry = self._get(federation_id)
        return load_config(entry.config_path)

    def save_config(self, federation_id: str, cfg: Dict) -> Dict:
        import yaml
        from src.config.schema import ConfigError, validate_config
        entry = self._get(federation_id)
        if entry.worker.running:
            return {"saved": False, "message": "Cannot update config while running"}
        try:
            validate_config(cfg)
        except ConfigError as e:
            return {"saved": False, "message": str(e)}
        with open(entry.config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        return {"saved": True, "path": str(entry.config_path)}

    def metrics(self, federation_id: str) -> List[Dict]:
        entry = self._get(federation_id)
        return entry.metrics_store.read_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, federation_id: str) -> FederationEntry:
        with self._lock:
            entry = self._federations.get(federation_id)
        if entry is None:
            raise KeyError(
                f"Federation '{federation_id}' not found. "
                f"Available: {list(self._federations.keys())}"
            )
        return entry

    def _write_example_configs(self) -> None:
        import yaml
        heart_cfg = {
            "model":       {"type": "kan", "params": {"layers_hidden": [13, 32, 16, 2], "grid_min": -2.0, "grid_max": 2.0, "num_grids": 8, "use_base_update": True}},
            "data":        {"type": "ehr", "params": {"data_source": "ucimlrepo", "batch_size": 32, "test_split": 0.2, "num_clients": 1, "client_id": 0}},
            "trainer":     {"type": "standard", "params": {"optimizer": "adam", "lr": 0.001, "momentum": 0.9}},
            "privacy":     {"type": "dpsgd", "params": {"noise_multiplier": 1.0, "clipping_norm": 1.0}},
            "compression": {"type": "topk", "params": {"top_k_ratio": 0.1}},
            "runtime":     {"device": "cpu", "server_address": "127.0.0.1:8080", "client_name": "hospital_01_heart"},
        }
        ecg_cfg = {
            "model":       {"type": "kan", "params": {"layers_hidden": [187, 64, 32, 5], "grid_min": -2.0, "grid_max": 2.0, "num_grids": 8, "use_base_update": True}},
            "data":        {"type": "ecg", "params": {"data_dir": "./data/mitbih", "batch_size": 64, "balance_classes": True, "max_samples_per_class": 5000, "num_clients": 1, "client_id": 0}},
            "trainer":     {"type": "standard", "params": {"optimizer": "adam", "lr": 0.001, "momentum": 0.9}},
            "privacy":     {"type": "dpsgd", "params": {"noise_multiplier": 0.8, "clipping_norm": 1.2}},
            "compression": {"type": "quantize", "params": {"bits": 16}},
            "runtime":     {"device": "cpu", "server_address": "127.0.0.1:8081", "client_name": "hospital_01_ecg"},
        }
        for name, cfg in [("heart_disease", heart_cfg), ("arrhythmia", ecg_cfg)]:
            path = self.federations_dir / f"{name}.yaml"
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)