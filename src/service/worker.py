"""
worker.py — FlowerWorker background process.

Uses multiprocessing.Process so stop() sends SIGTERM to actually
interrupt the blocking gRPC call.

Passes file paths (plain strings, picklable) to the subprocess.
The subprocess reconstructs StatusStore, MetricsStore, and
CheckpointStore from those paths so all writes go to the same
files the parent API reads.

Auto-resume: if a checkpoint exists, loads latest weights before
connecting to the server so training continues from where it stopped.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from pathlib import Path
from typing import Optional

from src.config.loader import load_config
from src.observerbility.checkpoint_store import CheckpointStore
from src.observerbility.metrics_store import MetricsStore
from src.observerbility.status_store import StatusStore

logger = logging.getLogger(__name__)


def _fl_process(
    config_path:       str,
    status_json_path:  str,
    metrics_csv_dir:   str,
    checkpoint_dir:    str,
    federation_id:     str,
    privacy_type:      str,
    noise_multiplier:  float,
    clipping_norm:     float,
    model_type:        str,
) -> None:
    """
    Target for the FL subprocess.

    Receives file PATHS as plain strings (picklable).
    Reconstructs all stores from those paths so writes go to the
    same files the parent process reads.
    """
    import flwr as fl
    from src.config.builder import build_from_config
    from src.config.loader import load_config
    from src.observerbility.checkpoint_store import CheckpointStore
    from src.observerbility.metrics_store import MetricsStore
    from src.observerbility.status_store import StatusStore

    status_store = StatusStore(path=status_json_path)

    metrics_store = MetricsStore(
        federation_id    = federation_id,
        output_dir       = metrics_csv_dir,
        privacy_type     = privacy_type,
        noise_multiplier = noise_multiplier,
        clipping_norm    = clipping_norm,
        model_type       = model_type,
    )

    checkpoint_store = CheckpointStore(
        federation_id  = federation_id,
        checkpoint_dir = checkpoint_dir,
        keep_last      = 5,
    )

    try:
        cfg   = load_config(config_path)
        built = build_from_config(
            cfg,
            status_store     = status_store,
            metrics_store    = metrics_store,
            checkpoint_store = checkpoint_store,
            federation_id    = federation_id,
        )

        # Auto-resume: load latest checkpoint if available
        resumed_round = checkpoint_store.load_latest(built.client.model)
        if resumed_round is not None:
            status_store.update(
                state   = "CONNECTING",
                message = f"Resuming from round {resumed_round} checkpoint...",
            )
        else:
            status_store.update(
                state   = "CONNECTING",
                message = "Connecting to FL server (fresh start)...",
            )

        fl.client.start_client(
            server_address = built.server_address,
            client         = built.client.to_client(),
        )

    except Exception as e:
        status_store.update(state="ERROR", message=f"Worker crashed: {e}")
        raise


class FlowerWorker:
    """
    Runs a Flower FL client in a background process.

    Uses multiprocessing.Process so stop() can send SIGTERM to
    actually interrupt the blocking gRPC call.
    """

    def __init__(
        self,
        config_path:      Path,
        status_store:     StatusStore,
        metrics_store:    Optional[MetricsStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        federation_id:    str = "default",
    ):
        self.config_path      = Path(config_path)
        self.status           = status_store
        self.metrics          = metrics_store
        self.checkpoint_store = checkpoint_store
        self.federation_id    = federation_id
        self._process:        Optional[multiprocessing.Process] = None
        self._monitor:        Optional[threading.Thread] = None
        self._running         = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            logger.warning("[%s] Worker already running", self.federation_id)
            return

        # Pre-read config to show metadata in UI before subprocess starts
        try:
            cfg              = load_config(self.config_path)
            privacy_type     = cfg.get("privacy",  {}).get("type",   "none")
            privacy_params   = cfg.get("privacy",  {}).get("params", {}) or {}
            model_type       = cfg.get("model",    {}).get("type",   "unknown")
            noise_multiplier = float(privacy_params.get("noise_multiplier", 0.0))
            clipping_norm    = float(privacy_params.get("clipping_norm",    0.0))
            self.status.update(
                state       = "CONNECTING",
                model       = model_type,
                privacy     = privacy_type,
                compression = cfg.get("compression", {}).get("type"),
                message     = "Starting worker...",
            )
        except Exception as e:
            logger.warning("[%s] Could not pre-read config: %s", self.federation_id, e)
            privacy_type, noise_multiplier, clipping_norm, model_type = "none", 0.0, 0.0, "unknown"

        # Resolve all file paths to pass as plain strings to subprocess
        status_json_path = str(Path(self.status.path).resolve())
        metrics_csv_dir  = str(
            Path(self.metrics.output_dir).resolve()
            if self.metrics else Path("output/metrics").resolve()
        )
        checkpoint_dir   = str(
            Path(self.checkpoint_store.checkpoint_dir).resolve()
            if self.checkpoint_store else Path("output/checkpoints").resolve()
        )

        self._process = multiprocessing.Process(
            target = _fl_process,
            args   = (
                str(self.config_path.resolve()),
                status_json_path,
                metrics_csv_dir,
                checkpoint_dir,
                self.federation_id,
                privacy_type,
                noise_multiplier,
                clipping_norm,
                model_type,
            ),
            daemon = True,
            name   = f"fl-process-{self.federation_id}",
        )
        self._process.start()
        self._running = True
        logger.info(
            "[%s] FL process started (pid=%d)", self.federation_id, self._process.pid
        )

        self._monitor = threading.Thread(
            target = self._watch,
            daemon = True,
            name   = f"fl-monitor-{self.federation_id}",
        )
        self._monitor.start()

    def stop(self) -> None:
        """Terminate the FL process — SIGTERM interrupts the blocking gRPC call."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
            logger.info("[%s] FL process terminated", self.federation_id)
            self.status.update(state="IDLE", message="Training stopped by user.")
        else:
            logger.warning("[%s] No running process to stop", self.federation_id)
        self._running = False

    def _watch(self) -> None:
        if self._process is None:
            return
        self._process.join()
        exit_code = self._process.exitcode

        if exit_code == 0:
            self.status.update(state="FINISHED", message="Training finished.")
            logger.info("[%s] FL process finished cleanly", self.federation_id)
        elif exit_code in (-15, -9):
            logger.info(
                "[%s] FL process terminated by user (exit=%d)",
                self.federation_id, exit_code,
            )
        else:
            msg = f"FL process exited with code {exit_code}"
            self.status.update(state="ERROR", message=msg)
            logger.error("[%s] %s", self.federation_id, msg)

        self._running = False