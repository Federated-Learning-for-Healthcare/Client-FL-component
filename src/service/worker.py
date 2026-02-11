from __future__ import annotations
from pathlib import Path
from typing import Optional
import threading
import flwr as fl

from src.config.loader import load_config
from src.config.builder import build_from_config
from src.observerbility.status_store import StatusStore


class FlowerWorker:
    def __init__(self, config_path: str | Path, status_store: StatusStore):
        self.config_path = Path(config_path)
        self.status = status_store
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._running = True
        try:
            cfg = load_config(self.config_path)

            # CONNECTING: worker-level phase
            self.status.update(
                state="CONNECTING",
                model=cfg.get("model", {}).get("type"),
                privacy=cfg.get("privacy", {}).get("type"),
                compression=cfg.get("compression", {}).get("type"),
                message="Connecting to FL server...",
            )

            # IMPORTANT: pass the SAME StatusStore into the built client
            built = build_from_config(cfg, status_store=self.status)

            fl.client.start_client(
                server_address=built.server_address,                
                client=built.client.to_client()
            )            

            self.status.update(state="FINISHED", message="Training finished (client disconnected).")

        except Exception as e:
            self.status.update(state="ERROR", message=f"Worker crashed: {e}")
        finally:
            self._running = False
