"""
app.py
Flask API for the AI-Powered Symptom-Based Disease Prediction System.

Endpoints:
  GET  /                      -> health check
  GET  /symptoms               -> list all valid symptoms (for Android dropdown/checklist UI)
  POST /predict                -> {"symptoms": ["itching","skin_rash"]} or {"text": "stomach ache, tired"}
  GET  /disease/<name>/info    -> description, precautions, medications, diet, workout, doctor for a disease
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


# ---------- Doctor recommendation mapping ----------
# Keys must exactly match the "Disease" values used in description_df / precautions_df
# (same casing/spelling your model outputs). If a disease is predicted but not listed
# here, DOCTOR_MAP.get() below falls back to "General Physician" instead of crashing.
DOCTOR_MAP = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist / Immunologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist / Gastroenterologist",
    "Drug Reaction": "Allergist / Immunologist",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedician",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist / Gastroenterologist",
    "Malaria": "General Physician",
    "Chicken pox": "Dermatologist / General Physician",
    "Dengue": "General Physician",
    "Typhoid": "General Physician",
    "hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemmorhoids(piles)": "Proctologist / General Surgeon",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthristis": "Orthopedician",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroymsal Positional Vertigo": "ENT Specialist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist",
}
DEFAULT_DOCTOR = "General Physician"


def get_recommended_doctor(disease_name: str) -> str:
    """Looks up the recommended specialist for a predicted disease.
    Falls back to General Physician if the disease isn't in the map,
    so this never throws even if a disease name doesn't match exactly."""
    return DOCTOR_MAP.get(disease_name, DEFAULT_DOCTOR)


def get_disease_info(disease_name: str) -> dict:
    """Pulls description, precautions, medications, diet, workout, and
    recommended doctor for one disease."""
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

    # NEW: recommended doctor specialty for this disease
    info["recommended_doctor"] = get_recommended_doctor(disease_name)

    return info


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "AI-Powered Symptom-Based Disease Prediction System",
        "endpoints": ["/symptoms", "/predict (POST)", "/disease/<name>/info", "/disease/<name>/doctor"]
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

    Returns prediction + top matches + full disease info (including recommended_doctor)
    for the top prediction.
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
        "recommended_doctor": disease_info["recommended_doctor"],  # NEW: also surfaced at top level
        "info": disease_info
    })


@app.route("/disease/<name>/info", methods=["GET"])
def disease_info_endpoint(name):
    info = get_disease_info(name)
    if info["description"] == "No description available.":
        return jsonify({"error": f"'{name}' not found. Check exact spelling/casing from a /predict response."}), 404
    return jsonify(info)


@app.route("/disease/<name>/doctor", methods=["GET"])
def disease_doctor_endpoint(name):
    """NEW: standalone lookup, handy for testing the mapping directly,
    e.g. GET /disease/Migraine/doctor"""
    return jsonify({"disease": name, "recommended_doctor": get_recommended_doctor(name)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
