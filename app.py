"""
app.py
Flask API for the AI-Powered Symptom-Based Disease Prediction System.

Endpoints:
  GET  /                      -> health check
  GET  /symptoms               -> list all valid symptoms (for Android dropdown/checklist UI)
  POST /predict                -> {"symptoms": ["itching","skin_rash"]} or {"text": "stomach ache, tired"}
  GET  /disease/<name>/info    -> description, precautions, medications, diet, workout for a disease
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import json

from predict import predict_disease, SYMPTOM_LIST
from symptom_matcher import normalize_symptoms, CLEAN_SYMPTOMS

app = Flask(__name__)
CORS(app)  # allow the Android app / any client to call this API

# ---------- Load recommendation datasets once at startup ----------
description_df = pd.read_csv("data/description.csv")
precautions_df = pd.read_csv("data/precautions_df.csv")
medications_df = pd.read_csv("data/medications.csv")
diets_df = pd.read_csv("data/diets.csv")
workout_df = pd.read_csv("data/workout_df.csv")


def get_disease_info(disease_name: str) -> dict:
    """Pulls description, precautions, medications, diet, and workout for one disease."""
    info = {"disease": disease_name}

    desc_row = description_df[description_df["Disease"] == disease_name]
    info["description"] = desc_row["Description"].values[0] if not desc_row.empty else "No description available."

    prec_row = precautions_df[precautions_df["Disease"] == disease_name]
    if not prec_row.empty:
        prec_cols = [c for c in precautions_df.columns if c.lower().startswith("precaution")]
        info["precautions"] = [
            prec_row[c].values[0] for c in prec_cols
            if pd.notna(prec_row[c].values[0])
        ]
    else:
        info["precautions"] = []

    med_row = medications_df[medications_df["Disease"] == disease_name]
    if not med_row.empty:
        raw = med_row["Medication"].values[0]
        try:
            info["medications"] = eval(raw) if isinstance(raw, str) and raw.startswith("[") else [raw]
        except Exception:
            info["medications"] = [raw]
    else:
        info["medications"] = []

    diet_row = diets_df[diets_df["Disease"] == disease_name]
    if not diet_row.empty:
        raw = diet_row["Diet"].values[0]
        try:
            info["diet"] = eval(raw) if isinstance(raw, str) and raw.startswith("[") else [raw]
        except Exception:
            info["diet"] = [raw]
    else:
        info["diet"] = []

    work_row = workout_df[workout_df["disease"] == disease_name] if "disease" in workout_df.columns else pd.DataFrame()
    if not work_row.empty and "workout" in workout_df.columns:
        info["workout"] = workout_df[workout_df["disease"] == disease_name]["workout"].tolist()
    else:
        info["workout"] = []

    return info


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "AI-Powered Symptom-Based Disease Prediction System",
        "endpoints": ["/symptoms", "/predict (POST)", "/disease/<name>/info"]
    })


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Returns the full list of valid symptoms in clean, human-readable form.
    Use this to populate the checklist/autocomplete in the Android app."""
    readable = sorted([s.replace("_", " ") for s in CLEAN_SYMPTOMS])
    return jsonify({"count": len(readable), "symptoms": readable})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts either:
      {"symptoms": ["itching", "skin_rash", "fatigue"]}   <- exact keys, from checkbox UI
      {"text": "stomach ache, throwing up, tired"}         <- free text, from voice input

    Returns prediction + top matches + full disease info for the top prediction.
    """
    body = request.get_json(force=True, silent=True) or {}

    if "text" in body and body["text"].strip():
        match_result = normalize_symptoms(body["text"])
        matched_symptoms = match_result["matched"]
        unmatched = match_result["unmatched"]
    elif "symptoms" in body and isinstance(body["symptoms"], list):
        match_result = normalize_symptoms(body["symptoms"])
        matched_symptoms = match_result["matched"]
        unmatched = match_result["unmatched"]
    else:
        return jsonify({"error": "Provide either 'text' (string) or 'symptoms' (list) in the request body."}), 400

    if not matched_symptoms:
        return jsonify({
            "error": "No recognizable symptoms found.",
            "unmatched": unmatched,
            "hint": "Try /symptoms to see the full list of valid symptom terms."
        }), 422

    result = predict_disease(matched_symptoms, top_n=3)
    disease_info = get_disease_info(result["prediction"])

    return jsonify({
        "matched_symptoms": matched_symptoms,
        "unmatched_input": unmatched,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top_matches": result["top_matches"],
        "info": disease_info
    })


@app.route("/disease/<name>/info", methods=["GET"])
def disease_info_endpoint(name):
    info = get_disease_info(name)
    if info["description"] == "No description available.":
        return jsonify({"error": f"'{name}' not found. Check exact spelling/casing from a /predict response."}), 404
    return jsonify(info)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
