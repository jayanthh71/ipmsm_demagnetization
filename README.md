# IPMSM Demagnetization Predictor

Runs a Simulink simulation of an IPMSM motor and predicts the demagnetization condition using a trained ML model.

## Requirements

- MATLAB R2025b
- Python 3.12

## Setup

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python train.py  # train the model
python run.py    # run a prediction
```
