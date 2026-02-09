# from __future__ import annotations
# import json
# import threading
# from pathlib import Path
# from typing import Any, Dict
# from datetime import datetime


# class StatusStore:
#     """
#     File-backed status store for UI polling.
#     Adds simple thread safety because Worker + Client can write concurrently.
#     """
#     def __init__(self, path: str | Path = "output/status.json"):
#         self.path = Path(path)
#         self.path.parent.mkdir(parents=True, exist_ok=True)
#         self._lock = threading.Lock()

#         # if not self.path.exists():
#         #     self.write({
#         #         "state": "IDLE",
#         #         "global_round": None,
#         #         "client_round": 0,
#         #         "train_loss": None,
#         #         "accuracy": None,
#         #         "eval_loss": None,
#         #         "history": [],

#         #         "model": None,
#         #         "privacy": None,
#         #         "compression": None,

#         #         "message": "Service started",
#         #         "last_update": self._now(),
#         #     })

#         if not self.path.exists():
#                 self.write({
#                     "state": "IDLE",
#                     "global_round": None,
#                     "client_round": 0,
                    
#                     "model": None,
#                     "privacy": None,
#                     "compression": None,

#                     "message": "Service started",
#                     "last_update": self._now(),
#                     "train_loss": None,
#                     "train_accuracy": None,
#                     "eval_loss": None,
#                     "eval_accuracy": None,
#                     "history": [],


#                 })
#     def _now(self) -> str:
#         return datetime.utcnow().isoformat()

#     def read(self) -> Dict[str, Any]:
#         with self._lock:
#             return json.loads(self.path.read_text(encoding="utf-8"))

#     def write(self, data: Dict[str, Any]) -> None:
#         data["last_update"] = self._now()
#         with self._lock:
#             self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

#     def update(self, **fields: Any) -> None:
#         """
#         Update current status. If metrics for a round are provided,
#         append to history for charting.
#         """
#         with self._lock:
#             current = json.loads(self.path.read_text(encoding="utf-8"))

#             # Update top-level fields
#             current.update(fields)
#             current["last_update"] = self._now()

#             # Append to history when round + metric info exists
#             client_round = current.get("client_round")
#             global_round = current.get("global_round")
#             # train_loss = current.get("train_loss")
#             # accuracy = current.get("accuracy")

#             # # Only append when we have a client_round and at least one metric
#             # if client_round is not None and (train_loss is not None or accuracy is not None):
#             #     hist = current.get("history", [])
#             #     hist.append({
#             #         "client_round": client_round,
#             #         "global_round": global_round,
#             #         "train_loss": train_loss,
#             #         "accuracy": accuracy,
#             #         "timestamp": current["last_update"],
#             #     })
#             #     # Keep last 100 points
#             #     current["history"] = hist[-100:]
#             train_loss = current.get("train_loss")
#             train_acc = current.get("train_accuracy")
#             eval_loss = current.get("eval_loss")
#             eval_acc = current.get("eval_accuracy")

#             # Only append history when TRAIN metrics exist (not eval)
#             if client_round is not None and (train_loss is not None or train_acc is not None):
#                 hist = current.get("history", [])
#                 hist.append({
#                     "client_round": client_round,
#                     "global_round": global_round,
#                     "train_loss": train_loss,
#                     "train_accuracy": train_acc,
#                     "eval_loss": eval_loss,
#                     "eval_accuracy": eval_acc,
#                     "timestamp": current["last_update"],
#                 })
#                 current["history"] = hist[-100:]  # Keep last 100 points


#             self.path.write_text(json.dumps(current, indent=2), encoding="utf-8")
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

        if not self.path.exists():
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

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

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
            current = json.loads(self.path.read_text(encoding="utf-8"))

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
