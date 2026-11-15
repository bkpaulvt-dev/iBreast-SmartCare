"""
iBreast SmartCare - research prototype
======================================
Browser app that takes four manually entered iBreast sensor readings,
computes the electrolyte ratios and applies the trained scikit-learn models
stored in ../models.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

The models to load, and the feature order each one was trained with, are
listed in models/models.json. Nothing about the models is hard-coded here.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CONFIG = MODELS_DIR / "models.json"

RAW_INPUTS = [
    # key, label, unit, default
    ("Na_mV", "Sodium electrode potential (Na⁺)", "mV", 0.0),
    ("K_mV", "Potassium electrode potential (K⁺)", "mV", 0.0),
    ("Cl_mV", "Chloride electrode potential (Cl⁻)", "mV", 0.0),
    ("EC_Ohm", "Impedance magnitude |Z| at 251 kHz", "Ω", 0.0),
]


def compute_features(v):
    """Four raw readings -> the ten features used in the analysis."""
    def ratio(a, b):
        return v[a] / v[b] if v[b] != 0 else float("nan")
    f = dict(v)
    f["Na_K_Ratio"] = ratio("Na_mV", "K_mV")
    f["Na_Cl_Ratio"] = ratio("Na_mV", "Cl_mV")
    f["K_Cl_Ratio"] = ratio("K_mV", "Cl_mV")
    f["Na_EC_Ratio"] = ratio("Na_mV", "EC_Ohm")
    f["K_EC_Ratio"] = ratio("K_mV", "EC_Ohm")
    f["Cl_EC_Ratio"] = ratio("Cl_mV", "EC_Ohm")
    return f


@st.cache_resource
def load_models():
    """Returns (list of loaded model entries, list of problems)."""
    loaded, problems = [], []
    if not CONFIG.exists():
        return loaded, [f"Missing {CONFIG.relative_to(ROOT)}"]
    for entry in json.loads(CONFIG.read_text()):
        path = MODELS_DIR / entry["file"]
        if not path.exists():
            problems.append(f"Model file not found: models/{entry['file']}")
            continue
        try:
            model = joblib.load(path)
        except Exception as exc:  # usually a scikit-learn version mismatch
            problems.append(f"Could not load models/{entry['file']}: {exc}")
            continue
        n_expected = getattr(model, "n_features_in_", None)
        if n_expected is not None and n_expected != len(entry["features"]):
            problems.append(
                f"models/{entry['file']} expects {n_expected} features but "
                f"models.json lists {len(entry['features'])}.")
            continue
        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != list(entry["features"]):
            problems.append(
                f"models/{entry['file']} was trained with features {list(names)} "
                f"but models.json lists {entry['features']}.")
            continue
        loaded.append({**entry, "model": model})
    return loaded, problems


st.set_page_config(page_title="iBreast SmartCare (research prototype)",
                   layout="centered")
st.title("iBreast SmartCare")
st.caption("Research prototype accompanying the iBreast manuscript")

st.warning(
    "**Research use only. Not a medical device.** The models were trained on "
    "milk from ten donors and are not clinically validated. The scores below "
    "are uncalibrated classifier outputs, not disease probabilities, and must "
    "not be used for diagnosis or treatment decisions.")

st.subheader("Sensor readings")
st.write("Enter the four values read from the iBreast sensor read-out. "
         "Values are entered manually; there is no automatic link to the sensor.")

values = {}
cols = st.columns(2)
for i, (key, label, unit, default) in enumerate(RAW_INPUTS):
    values[key] = cols[i % 2].number_input(
        f"{label} ({unit})", value=default, step=0.1, format="%.3f", key=key)

models, problems = load_models()
for p in problems:
    st.error(p)

if st.button("Compute classifier scores", type="primary"):
    if any(values[k] == 0 for k, *_ in RAW_INPUTS):
        st.error("Enter a non-zero value for all four readings.")
    elif not models:
        st.error("No models are available. See the messages above.")
    else:
        feats = compute_features(values)
        st.subheader("Derived features")
        shown = ["Na_K_Ratio", "Na_Cl_Ratio", "K_Cl_Ratio",
                 "Na_EC_Ratio", "K_EC_Ratio", "Cl_EC_Ratio"]
        st.dataframe(pd.DataFrame(
            {"Feature": shown, "Value": [round(feats[k], 4) for k in shown]}),
            hide_index=True)

        st.subheader("Classifier outputs")
        rows = []
        for m in models:
            x = pd.DataFrame([[feats[f] for f in m["features"]]],
                             columns=m["features"])
            if getattr(m["model"], "feature_names_in_", None) is None:
                x = x.to_numpy()
            try:
                score = float(m["model"].predict_proba(x)[0, 1])
                label = int(m["model"].predict(x)[0])
            except Exception as exc:
                st.error(f"{m['name']}: prediction failed ({exc})")
                continue
            rows.append({
                "Condition": m["name"],
                "Algorithm": m.get("algorithm", ""),
                "Input features": ", ".join(m["features"]),
                "Classifier score (0-1)": round(score, 3),
                "Predicted class": "positive" if label == 1 else "negative",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(
                "Classifier score = the model's predict_proba output for the "
                "positive class. It is not calibrated and is not a risk estimate. "
                "For the SVM, the score and the predicted class are computed "
                "differently and can disagree.")

st.divider()
st.caption("Code, trained models and data: see the repository README. "
           "Labels used for training were self-reported by participants.")
