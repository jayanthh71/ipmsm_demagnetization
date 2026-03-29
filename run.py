import json
from pathlib import Path
import joblib
import matlab.engine
import pandas as pd

SIMULATION_DIR = Path(__file__).parent / "simulation"
PIPELINE = joblib.load(Path(__file__).parent / "demag_regression_pipeline.pkl")
FEATURES = list(PIPELINE.feature_names_in_)


def classify_demag(psim):
    demag_pct = (0.04366 - psim) / 0.04366 * 100
    if demag_pct < 10:
        label = "Healthy"
    elif demag_pct < 20:
        label = "Mild Demagnetization"
    elif demag_pct < 35:
        label = "Moderate Demagnetization"
    else:
        label = "Severe Demagnetization"
    return label, demag_pct


def predict_from_dict(input_dict):
    df = pd.DataFrame([input_dict], columns=FEATURES)

    psim_pred = PIPELINE.predict(df)[0]

    label, demag_pct = classify_demag(psim_pred)

    return {
        "flux_linkage_Wb": float(psim_pred),
        "demagnetization_%": float(demag_pct),
        "condition": label,
    }


def run_simulation(
    eng: matlab.engine.MatlabEngine, speed: float, torque: float, psim: float
) -> dict:
    eng.workspace["input_speed"] = speed
    eng.workspace["input_torque"] = torque
    eng.workspace["input_psim"] = psim
    eng.cd(str(SIMULATION_DIR), nargout=0)

    print("Running MATLAB simulation...")
    eng.run("Test.m", nargout=0)

    with open(SIMULATION_DIR / "features.json") as f:
        features = json.load(f)

    print("Running prediction...")
    return predict_from_dict(features)


if __name__ == "__main__":
    speed = float(input("Rotor speed (RPM): "))
    torque = float(input("Load torque (Nm): "))
    psim = float(input("Flux linkage (Wb): "))

    print("Starting MATLAB engine...")
    eng = matlab.engine.start_matlab()

    result = run_simulation(eng, speed, torque, psim)

    eng.quit()

    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"{k:25s}: {v}")
