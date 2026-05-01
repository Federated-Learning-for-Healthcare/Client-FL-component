"""
checkpoint_store.py — Per-federation model checkpoint manager.

Lock discipline:
  _sorted_paths() NEVER acquires self._lock — it is a pure filesystem
  scan with no shared state. This makes it safe to call both inside
  and outside locked sections without risk of deadlock or redundancy.

  All methods that call torch.load() do so inside self._lock.
  list_checkpoints() snapshots the path list under one lock acquisition,
  then loads each file individually under separate acquisitions so it
  does not block saves across the whole loop.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointStore:

    def __init__(self, federation_id: str, checkpoint_dir: str, keep_last: int = 5):
        self.federation_id  = federation_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last      = keep_last
        self._lock          = threading.Lock()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointStore [%s]: dir=%s  keep_last=%s",
                    federation_id, self.checkpoint_dir, keep_last)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, model: nn.Module, global_round: int) -> Path:
        filename = self._filename(global_round)
        with self._lock:
            torch.save({"global_round": global_round,
                        "federation_id": self.federation_id,
                        "state_dict": model.state_dict()}, filename)
        logger.info("CheckpointStore [%s]: saved round %d → %s",
                    self.federation_id, global_round, filename.name)
        if self.keep_last is not None:
            self._prune()
        return filename

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_latest(self, model: nn.Module) -> Optional[int]:
        """Load most recent checkpoint into model. Returns round number or None."""
        # _sorted_paths() does NOT hold the lock — safe to call inside here
        with self._lock:
            paths = self._sorted_paths()
            if not paths:
                logger.info("CheckpointStore [%s]: no checkpoint found", self.federation_id)
                return None
            path = paths[-1]
            data = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(data["state_dict"])
            round_num = data.get("global_round", 0)
        logger.info("CheckpointStore [%s]: loaded round %d from %s",
                    self.federation_id, round_num, path.name)
        return round_num

    def load_round(self, model: nn.Module, global_round: int) -> bool:
        path = self._filename(global_round)
        if not path.exists():
            return False
        with self._lock:
            data = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(data["state_dict"])
        return True

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def latest_round(self) -> Optional[int]:
        # _sorted_paths() does NOT hold the lock — safe to call inside here
        with self._lock:
            paths = self._sorted_paths()
            if not paths:
                return None
            data = torch.load(paths[-1], map_location="cpu", weights_only=True)
            return data.get("global_round")

    def list_checkpoints(self) -> List[Dict]:
        """
        Snapshot the sorted path list in one lock acquisition, then load
        each file individually. Does not hold the lock across the whole
        loop so saves are never blocked for the full duration.
        """
        # Snapshot path list — _sorted_paths() doesn't need the lock
        # but we hold it here to serialise against concurrent saves that
        # may be adding new files.
        with self._lock:
            paths = list(reversed(self._sorted_paths()))  # newest first

        result = []
        for path in paths:
            try:
                with self._lock:
                    data = torch.load(path, map_location="cpu", weights_only=True)
                result.append({"filename": path.name,
                               "global_round": data.get("global_round"),
                               "size_bytes": path.stat().st_size})
            except Exception as e:
                logger.warning("CheckpointStore: could not read %s: %s", path, e)
        return result

    def has_checkpoint(self) -> bool:
        return bool(list(self.checkpoint_dir.glob("round_*.pt")))

    # ------------------------------------------------------------------
    # Internal
    # _sorted_paths() is intentionally lock-free — it only does a
    # filesystem glob and a sort, with no access to shared mutable state.
    # This means it is safe to call from inside OR outside a locked
    # section without risk of deadlock or double-acquisition.
    # ------------------------------------------------------------------

    def _filename(self, r: int) -> Path:
        return self.checkpoint_dir / f"round_{r:05d}.pt"

    def _sorted_paths(self) -> List[Path]:
        """Return checkpoint paths sorted by round number (ascending).
        Parses round number from filename — independent of mtime."""
        paths = list(self.checkpoint_dir.glob("round_*.pt"))
        paths.sort(key=lambda p: int(p.stem.split("_")[1]))
        return paths

    def _prune(self) -> None:
        paths = self._sorted_paths()   # no lock needed here
        for path in paths[: max(0, len(paths) - self.keep_last)]:
            try:
                path.unlink()
                logger.debug("CheckpointStore [%s]: pruned %s",
                             self.federation_id, path.name)
            except Exception as e:
                logger.warning("CheckpointStore prune failed %s: %s", path.name, e)