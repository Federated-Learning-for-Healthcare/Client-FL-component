"""
inference_engine.py — Per-federation inference engine.

Fix: model_type and data_type are now cached after the first
_build_and_load() call. load_config() is no longer called on every
predict() invocation — only when the model is actually rebuilt.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config.loader import load_config
from src.config.registry import DEFAULT_REGISTRY
from src.observerbility.checkpoint_store import CheckpointStore

logger = logging.getLogger(__name__)

_CLASS_LABELS = {
    "ehr":   {0: "No disease",         1: "Heart disease"},
    "ecg":   {0: "Normal",             1: "Supraventricular",
               2: "Ventricular",        3: "Fusion",
               4: "Unclassifiable"},
    "mnist": {i: str(i) for i in range(10)},
}


class InferenceEngine:
    """
    Loads the latest checkpoint and runs predictions, completely
    independent of the training subprocess.

    Thread-safe. Auto-reloads when a newer checkpoint is available.
    Config is read once on first load and cached — not on every call.
    """

    def __init__(
        self,
        federation_id:    str,
        config_path:      Path,
        checkpoint_store: CheckpointStore,
        device:           str = "cpu",
    ):
        self.federation_id    = federation_id
        self.config_path      = Path(config_path)
        self.checkpoint_store = checkpoint_store
        self.device           = device
        self._lock            = threading.Lock()

        # Cached after first _build_and_load() — never re-read from YAML
        self._model:        Optional[Any] = None
        self._loaded_round: Optional[int] = None
        self._model_type:   Optional[str] = None   # cached
        self._data_type:    Optional[str] = None   # cached

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, inputs: List[List[float]]) -> Dict[str, Any]:
        if not self.checkpoint_store.has_checkpoint():
            return {
                "error":         "no_checkpoint",
                "message":       "No checkpoint available yet. Start training first.",
                "federation_id": self.federation_id,
            }

        model, loaded_round, model_type, data_type = self._get_model()

        try:
            x = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        except Exception as e:
            return {"error": "invalid_input", "message": str(e)}

        with torch.no_grad():
            logits = model(x)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()

        class_labels = _CLASS_LABELS.get(data_type, {})
        predictions  = []
        for prob_row in probs:
            label      = int(np.argmax(prob_row))
            confidence = float(prob_row[label])
            predictions.append({
                "label":         label,
                "label_name":    class_labels.get(label, str(label)),
                "confidence":    round(confidence, 4),
                "probabilities": {
                    class_labels.get(i, str(i)): round(float(p), 4)
                    for i, p in enumerate(prob_row)
                },
            })

        return {
            "federation_id": self.federation_id,
            "global_round":  loaded_round,
            "model_type":    model_type,
            "data_type":     data_type,
            "predictions":   predictions,
        }

    def latest_round(self) -> Optional[int]:
        return self._loaded_round or self.checkpoint_store.latest_round()

    def is_ready(self) -> bool:
        return self.checkpoint_store.has_checkpoint()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model(self):
        """
        Return (model, loaded_round, model_type, data_type).

        Rebuilds the model only when a newer checkpoint is available.
        model_type and data_type are cached after the first build so
        load_config() is never called on subsequent predict() calls.
        """
        with self._lock:
            latest = self.checkpoint_store.latest_round()

            if self._model is None or latest != self._loaded_round:
                (self._model,
                 self._loaded_round,
                 self._model_type,
                 self._data_type) = self._build_and_load(latest)

            return (self._model, self._loaded_round,
                    self._model_type, self._data_type)

    def _build_and_load(self, round_num: Optional[int]):
        """
        Reconstruct model from config and load checkpoint weights.
        Called only when the model needs to be rebuilt — not on every
        prediction. model_type and data_type are returned and cached
        by the caller.
        """
        cfg          = load_config(self.config_path)   # read once here only
        model_type   = cfg["model"]["type"]
        data_type    = cfg["data"]["type"]
        model_params = cfg["model"].get("params", {}) or {}

        model_cls = DEFAULT_REGISTRY.models.get(model_type)
        if model_cls is None:
            raise ValueError(
                f"Unknown model type '{model_type}' in {self.config_path}"
            )

        try:
            model = model_cls(**model_params) if model_params else model_cls()
        except TypeError as e:
            raise ValueError(
                f"Failed to build model '{model_type}' "
                f"with params {model_params}: {e}"
            ) from e

        model.to(self.device)
        model.eval()

        loaded_round = self.checkpoint_store.load_latest(model)

        logger.info(
            "InferenceEngine [%s]: built model=%s  data=%s  round=%s",
            self.federation_id, model_type, data_type, loaded_round,
        )
        return model, loaded_round, model_type, data_type