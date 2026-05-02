from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


class StatusStore:
    """
    File-backed status store for UI polling.
    Thread-safe because Worker + Client can write concurrently.

    This version separates TRAIN vs EVAL metrics so your charts don't get polluted.
    """
    def __init__(self, path: str | Path = "output/status.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        if not self._has_valid_json():
            self.write({
                "state": "IDLE",
                "global_round": None,
                "client_round": 0,

                "model": None,
                "privacy": None,
                "compression": None,

                "message": "Service started",
                "last_update": self._now(),

                # latest TRAIN metrics
                "train_loss": None,
                "train_accuracy": None,

                # latest EVAL metrics
                "eval_loss": None,
                "eval_accuracy": None,

                # separate histories (IMPORTANT)
                "train_history": [],
                "eval_history": [],
            })

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _has_valid_json(self) -> bool:
        if not self.path.exists():
            return False
        try:
            text = self.path.read_text(encoding="utf-8").strip()
            if not text:
                return False
            json.loads(text)
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def _default_state(self) -> Dict[str, Any]:
        return {
            "state": "IDLE", "global_round": None, "client_round": 0,
            "model": None, "privacy": None, "compression": None,
            "message": "Service started", "last_update": self._now(),
            "train_loss": None, "train_accuracy": None,
            "eval_loss": None, "eval_accuracy": None,
            "train_history": [], "eval_history": [],
        }

    def read(self) -> Dict[str, Any]:
        with self._lock:
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return self._default_state()

    def write(self, data: Dict[str, Any]) -> None:
        data["last_update"] = self._now()
        with self._lock:
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update(self, **fields: Any) -> None:
        """
        Update current status.

        - If train_loss/train_accuracy updated -> append to train_history
        - If eval_loss/eval_accuracy updated -> append to eval_history

        This prevents eval values from being copied into every train row.
        """
        with self._lock:
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                current = self._default_state()

            # Detect what kind of update this is BEFORE we overwrite fields
            train_update = ("train_loss" in fields) or ("train_accuracy" in fields)
            eval_update = ("eval_loss" in fields) or ("eval_accuracy" in fields)

            # Update top-level fields
            current.update(fields)
            current["last_update"] = self._now()

            client_round = current.get("client_round")
            global_round = current.get("global_round")
            ts = current["last_update"]

            # Append TRAIN history point only when train metrics were updated
            if train_update and client_round is not None:
                hist = current.get("train_history", [])
                hist.append({
                    "client_round": client_round,
                    "global_round": global_round,
                    "train_loss": current.get("train_loss"),
                    "train_accuracy": current.get("train_accuracy"),
                    "timestamp": ts,
                })
                current["train_history"] = hist[-200:]  # keep last N points

            # Append EVAL history point only when eval metrics were updated
            if eval_update and client_round is not None:
                eh = current.get("eval_history", [])
                eh.append({
                    "client_round": client_round,
                    "global_round": global_round,
                    "eval_loss": current.get("eval_loss"),
                    "eval_accuracy": current.get("eval_accuracy"),
                    "timestamp": ts,
                })
                current["eval_history"] = eh[-200:]

            self.path.write_text(json.dumps(current, indent=2), encoding="utf-8")
