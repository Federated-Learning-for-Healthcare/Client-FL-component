from __future__ import annotations
import json
import threading
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


class StatusStore:
    """
    File-backed status store for UI polling.
    Adds simple thread safety because Worker + Client can write concurrently.
    """
    def __init__(self, path: str | Path = "output/status.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        if not self.path.exists():
            self.write({
                "state": "IDLE",
                "global_round": None,
                "client_round": 0,
                "train_loss": None,
                "accuracy": None,
                "eval_loss": None,
                "history": [],

                "model": None,
                "privacy": None,
                "compression": None,

                "message": "Service started",
                "last_update": self._now(),
            })

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, data: Dict[str, Any]) -> None:
        data["last_update"] = self._now()
        with self._lock:
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def update(self, **fields: Any) -> None:
        """
        Update current status. If metrics for a round are provided,
        append to history for charting.
        """
        with self._lock:
            current = json.loads(self.path.read_text(encoding="utf-8"))

            # Update top-level fields
            current.update(fields)
            current["last_update"] = self._now()

            # Append to history when round + metric info exists
            client_round = current.get("client_round")
            global_round = current.get("global_round")
            train_loss = current.get("train_loss")
            accuracy = current.get("accuracy")

            # Only append when we have a client_round and at least one metric
            if client_round is not None and (train_loss is not None or accuracy is not None):
                hist = current.get("history", [])
                hist.append({
                    "client_round": client_round,
                    "global_round": global_round,
                    "train_loss": train_loss,
                    "accuracy": accuracy,
                    "timestamp": current["last_update"],
                })
                # Keep last 100 points
                current["history"] = hist[-100:]

            self.path.write_text(json.dumps(current, indent=2), encoding="utf-8")
