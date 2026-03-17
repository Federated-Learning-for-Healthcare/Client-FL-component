from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI
from src.observerbility.status_store import StatusStore
from src.service.worker import FlowerWorker
from src.config.loader import load_config

#for static file creation (UI)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hospital FL Client Service", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[2]

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")




#BASE_DIR = Path(__file__).resolve().parents[2]  # project root
CONFIG_PATH = BASE_DIR / "conf" / "config.yaml"

status_store = StatusStore(path=str(BASE_DIR / "output" / "status.json"))
worker = FlowerWorker(config_path=CONFIG_PATH, status_store=status_store)


@app.get("/ui/config")
def ui_config(request: Request):
    return templates.TemplateResponse("config.html", {"request": request, "active": "config", "title": "Config"})

@app.get("/ui/status")
def ui_status(request: Request):
    return templates.TemplateResponse("status.html", {"request": request, "active": "status", "title": "Status"})



@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    data = status_store.read()
    data["worker_running"] = worker.running
    return data


@app.get("/config")
def config():
    cfg = load_config(CONFIG_PATH)
    # Return only the important parts (safe + readable)
    return {
        "model": cfg.get("model", {}),
        "data": cfg.get("data", {}),
        "trainer": {
            "type": cfg.get("trainer", {}).get("type"),
            "params": {k: v for k, v in cfg.get("trainer", {}).get("params", {}).items() if k != "local_epochs"},
        },
        "privacy": cfg.get("privacy", {}),
        "compression": cfg.get("compression", {}),
        "runtime": cfg.get("runtime", {}),
    }


@app.post("/start-training")
def start_training():
    if worker.running:
        return {"started": False, "message": "Worker already running"}
    worker.start()
    return {"started": True, "message": "Worker started"}
