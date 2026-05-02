# Healthcare Federated Learning Framework

A modular federated learning client for healthcare data, built on [Flower](https://flower.ai). Designed to run on a hospital node alongside existing systems, supporting multiple concurrent FL federations with differential privacy and model compression.

**Phase 1** — Client node implementation (ECG arrhythmia classification + EHR heart disease prediction).

---

## Architecture Overview

```
Hospital Node
├── FastAPI Service (port 8000)        ← you control this
│   ├── FederationManager
│   │   ├── Federation: Arrhythmia     ← connects to FL server :8081
│   │   └── Federation: Heart_disease  ← connects to FL server :8080
│   └── InferenceEngine (per federation)
│
FL Server (separate process / remote)  ← must be running before you start training
├── Arrhythmia server   :8081
└── Heart_disease server :8080
```

This repo is the **client node**. You need a Flower server running separately for training to start. Inference works from saved checkpoints without a server.

---

## Prerequisites

- Python **3.10 – 3.11** (3.12+ not tested)
- pip
- Git

> **NumPy version is critical.** `numpy<2.0.0` is enforced in requirements. Flower breaks silently with NumPy 2.x.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd healthcare_fl_framework
```

### 2. Create a virtual environment

```bash
# Create
python -m venv venv

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the install

```bash
python -c "import flwr, torch, numpy; print('flwr', flwr.__version__, '| torch', torch.__version__, '| numpy', numpy.__version__)"
```

Expected output: numpy version should be `1.x.x`, not `2.x.x`.

---

## Dataset Setup

### ECG — MIT-BIH Arrhythmia (manual download required)

1. Download the **Kaggle ECG Heartbeat Categorization** dataset:
   [https://www.kaggle.com/datasets/shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

2. Place the two CSV files here:

```
data/
└── mitbih/
    ├── mitbih_train.csv   ← 87,554 samples
    └── mitbih_test.csv    ← 21,892 samples
```

### EHR — Cleveland Heart Disease (auto-downloaded)

No action needed. The `ucimlrepo` package fetches the UCI Cleveland dataset automatically on first training run. Requires an internet connection on first use only.

Alternatively, download manually and set `data_source: csv` in `conf/federations/Heart_disease.yaml`:

```
data/
└── heart.csv
```

---

## Configuration

Federation configs live in `conf/federations/`. Each YAML file is auto-discovered as a separate federation.

```
conf/
└── federations/
    ├── Arrhythmia.yaml       ← ECG federation  (KAN model, DP-SGD, quantization)
    └── Heart_disease.yaml    ← EHR federation  (KAN model, DP-SGD, top-k compression)
```

Key settings to change before running:

```yaml
# In each federation YAML
runtime:
  server_address: "127.0.0.1:8080"   # ← Change to your FL server address
  client_name: "hospital_01"          # ← Identify this node
  device: "cpu"                       # ← Change to "cuda" if GPU available
```

---

## Running the Service

### Start the FastAPI client service

```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at `http://localhost:8000`.

Dashboard UI: `http://localhost:8000/ui`

API docs (Swagger): `http://localhost:8000/docs`

For production (no auto-reload):

```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Running the FL Server

You need a Flower server running before you can start training. The server aggregates model updates from all hospital clients.

### Minimal server script

Create `run_server.py` in your server machine (or locally for testing):

```python
import flwr as fl

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
)
```

Run it:

```bash
python run_server.py
```

Run a second instance on port 8081 for the Arrhythmia federation.

> The FL server and client service are fully independent — start the server first, then start training via the API.

---

## API Quick Reference

### Check status

```bash
# All federations
curl http://localhost:8000/federations

# Single federation
curl http://localhost:8000/federations/Arrhythmia/status
curl http://localhost:8000/federations/Heart_disease/status
```

### Start / stop training

```bash
# Start one federation (FL server must be running first)
curl -X POST http://localhost:8000/federations/Arrhythmia/start
curl -X POST http://localhost:8000/federations/Heart_disease/start

# Start all federations at once
curl -X POST http://localhost:8000/start-all

# Stop training
curl -X POST http://localhost:8000/federations/Arrhythmia/stop
curl -X POST http://localhost:8000/stop-all
```

### View training metrics

```bash
curl http://localhost:8000/federations/Arrhythmia/metrics
curl http://localhost:8000/federations/Heart_disease/metrics
```

Metrics are also written to CSV after every round:

```
output/metrics/Arrhythmia_metrics.csv
output/metrics/Heart_disease_metrics.csv
```

### Run inference (requires at least one completed training round)

```bash
# ECG — input is 187 time-step signal values
curl -X POST http://localhost:8000/federations/Arrhythmia/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[0.1, 0.2, 0.0, ...]]}'

# EHR — input is 13 clinical features (Cleveland format)
curl -X POST http://localhost:8000/federations/Heart_disease/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]}'
```

Response includes label name, confidence score, and full probability distribution.

### Check inference readiness

```bash
curl http://localhost:8000/federations/Arrhythmia/inference/status
```

### Update config without restarting the service

```bash
curl -X POST http://localhost:8000/federations/Heart_disease/config \
  -H "Content-Type: application/json" \
  -d @conf/federations/Heart_disease.yaml
```

---

## Output Files

```
output/
├── fed_Arrhythmia/
│   ├── status.json              ← live training status (updated each round)
│   └── checkpoints/
│       ├── round_00001.pt       ← model checkpoint (keep last 5)
│       └── round_00005.pt
├── fed_Heart_disease/
│   ├── status.json
│   └── checkpoints/
└── metrics/
    ├── Arrhythmia_metrics.csv   ← per-round: loss, accuracy, compression ratio, privacy params
    └── Heart_disease_metrics.csv
```

Checkpoints persist across restarts. If training is stopped and restarted, the client auto-resumes from the latest checkpoint before reconnecting to the server.

---

## Project Structure

```
healthcare_fl_framework/
├── conf/
│   └── federations/             # Per-federation YAML configs (auto-discovered)
├── data/                        # Datasets (gitignored, must populate manually)
├── output/                      # Training outputs (gitignored)
├── src/
│   ├── config/
│   │   ├── builder.py           # Builds all modules from a config dict
│   │   ├── loader.py            # YAML loading
│   │   ├── registry.py          # Maps type strings → classes
│   │   └── schema.py            # Config validation
│   ├── core/
│   │   ├── client.py            # ModularFlowerClient (Flower NumPyClient)
│   │   └── interfaces.py        # Abstract base classes for all modules
│   ├── dataLoaders/
│   │   ├── ecg_loader.py        # MIT-BIH ECG (187 features, 5 classes)
│   │   ├── ehr_loader.py        # Cleveland EHR (13 features, binary)
│   │   └── mnist_loader.py      # MNIST (dev/testing only)
│   ├── models/
│   │   ├── kan.py               # Kolmogorov-Arnold Network (FastKAN)
│   │   └── mlp.py               # SimpleMLP baseline
│   ├── modules/
│   │   ├── compression.py       # NoCompression, TopK, Quantization
│   │   ├── privacy.py           # NoPrivacy, GaussianPrivacy, DPSGDPrivacy
│   │   └── training.py          # StandardPyTorchTrainer (SGD / Adam)
│   ├── observerbility/
│   │   ├── checkpoint_store.py  # Model checkpoint save/load (keep_last=5)
│   │   ├── metrics_store.py     # Per-round CSV metrics
│   │   └── status_store.py      # Live JSON status
│   └── service/
│       ├── app.py               # FastAPI app + all REST endpoints
│       ├── federation_manager.py# Manages N concurrent federations
│       ├── inference_engine.py  # Loads checkpoint, runs predictions
│       └── worker.py            # Flower client subprocess wrapper
├── templates/                   # Jinja2 dashboard templates (temporary)
├── requirements.txt
└── user_app.py                  # Legacy single-federation CLI entry point
```

---

## Adding a New Federation

1. Create a new YAML in `conf/federations/MyFederation.yaml` using an existing config as a template.
2. Restart the FastAPI service — it auto-discovers all YAML files on startup.
3. Start the corresponding FL server on the configured port.
4. Start training: `POST /federations/MyFederation/start`

---

## Plugin Registry

The framework is fully modular. Supported types in YAML configs:

| Category | `type` value | Class |
|---|---|---|
| model | `kan` | KAN (Kolmogorov-Arnold Network) |
| model | `mlp` | SimpleMLP |
| trainer | `standard` | StandardPyTorchTrainer |
| privacy | `none` | NoPrivacy |
| privacy | `gaussian` | GaussianPrivacy |
| privacy | `dpsgd` | DPSGDPrivacy (Abadi et al. 2016) |
| compression | `none` | NoCompression |
| compression | `topk` | TopKCompression |
| compression | `quantize` | QuantizationCompression |
| data | `mnist` | MNISTDataLoader |
| data | `ehr` | EHRLoader |
| data | `ecg` | ECGLoader |

---

## Troubleshooting

**Training starts but immediately goes to ERROR**
- FL server is not running, or `server_address` in the federation YAML is wrong.
- Check: `curl http://localhost:8000/federations/<id>/status`

**`numpy` version conflict on install**
- Run `pip install "numpy<2.0.0"` explicitly after installing requirements.

**ECG federation errors on start**
- MIT-BIH CSV files are missing. Download from Kaggle and place in `data/mitbih/`.

**Heart disease federation errors on first run**
- `ucimlrepo` download failed (no internet). Download `heart.csv` manually and set `data_source: csv` in `Heart_disease.yaml`.

**Inference returns `{"error": "no_checkpoint"}`**
- No training round has completed yet. Start the FL server, start training, wait for at least one round to finish.

**Port 8000 already in use**
```bash
uvicorn src.service.app:app --host 0.0.0.0 --port 8001
```

---

## Tech Stack

| Component | Library |
|---|---|
| Federated Learning | [Flower](https://flower.ai) >= 1.13.0 |
| Neural Networks | PyTorch |
| API Service | FastAPI + Uvicorn |
| Config | YAML + OmegaConf |
| Privacy | DP-SGD (custom, Abadi et al. 2016) |
| Data | pandas, scikit-learn, ucimlrepo |
| Monitoring | psutil |
