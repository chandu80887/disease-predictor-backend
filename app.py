"""
app.py
Flask API for the AI-Powered Symptom-Based Disease Prediction System.

Endpoints:
  GET  /                      -> health check
  GET  /symptoms               -> list all valid symptoms (for Android dropdown/checklist UI)
  POST /predict                -> {"symptoms": ["itching","skin_rash"]} or {"text": "stomach ache, tired"}
  GET  /disease/<name>/info    -> description, precautions, medications, diet, workout, doctor for a disease
  GET  /disease/<name>/doctor  -> standalone doctor lookup for a disease
  GET  /hospitals              -> nearby hospitals sorted by specialty match + distance
"""

import math

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


# ---------- Load translations for Hindi and Kannada ----------
with open("translations.json", "r", encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)


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
    "Jaundice": "Hepatologist",
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
        "endpoints": ["/symptoms", "/predict (POST)", "/disease/<name>/info", "/disease/<name>/doctor", "/hospitals"]
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
      {"symptoms": ["itching", "skin_rash", "fatigue"], "language": "hi"}   <- exact keys, from checkbox UI
      {"text": "stomach ache, throwing up, tired", "language": "kn"}         <- free text, from voice input

    Language options: "en" (English), "hi" (Hindi), "kn" (Kannada)
    Defaults to "en" if not specified.

    Returns prediction in both English and localized language.
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

    # Get language from request (default to English)
    language = body.get("language", "en")
    
    # Get localized names for disease and doctor
    disease_name_en = result["prediction"]
    disease_name_local = TRANSLATIONS["diseases"].get(disease_name_en, {}).get(language, disease_name_en)
    
    doctor_name_en = disease_info["recommended_doctor"]
    doctor_name_local = TRANSLATIONS["doctors"].get(doctor_name_en, {}).get(language, doctor_name_en)

    return jsonify({
        "matched_symptoms": matched_symptoms,
        "unmatched_input": unmatched,
        "prediction": result["prediction"],
        "prediction_local": disease_name_local,
        "confidence": result["confidence"],
        "top_matches": result["top_matches"],
        "recommended_doctor": disease_info["recommended_doctor"],
        "recommended_doctor_local": doctor_name_local,
        "language": language,
        "info": disease_info
    })


@app.route("/disease/<name>/info", methods=["GET"])
def disease_info_endpoint(name):
    info = get_disease_info(name)
    
    # Add localized disease and doctor names
    language = request.args.get("language", "en")
    disease_local = TRANSLATIONS["diseases"].get(name, {}).get(language, name)
    doctor_local = TRANSLATIONS["doctors"].get(info.get("recommended_doctor", ""), {}).get(language, info.get("recommended_doctor", ""))
    
    info["disease_local"] = disease_local
    info["recommended_doctor_local"] = doctor_local
    info["language"] = language
    
    if info["description"] == "No description available.":
        return jsonify({"error": f"'{name}' not found. Check exact spelling/casing from a /predict response."}), 404
    return jsonify(info)


@app.route("/disease/<name>/doctor", methods=["GET"])
def disease_doctor_endpoint(name):
    """NEW: standalone lookup, handy for testing the mapping directly,
    e.g. GET /disease/Migraine/doctor"""
    doctor_en = get_recommended_doctor(name)
    
    # Add localized doctor name
    language = request.args.get("language", "en")
    doctor_local = TRANSLATIONS["doctors"].get(doctor_en, {}).get(language, doctor_en)
    
    return jsonify({
        "disease": name,
        "recommended_doctor": doctor_en,
        "recommended_doctor_local": doctor_local,
        "language": language
    })


# ============================================================
# NEARBY HOSPITAL LOCATOR MODULE
# ============================================================

# ============================================================
# MYSURU HOSPITAL / HEALTH CENTRE DATABASE
# 100+ hospitals and health centres
# ============================================================

HOSPITAL_LIST = [

    # ========================================================
    # MAJOR GOVERNMENT HOSPITALS
    # ========================================================

    {
        "name": "Krishna Rajendra Hospital (KR Hospital)",
        "type": "Government",
        "address": "Irwin Road, Devaraja Mohalla, Mysuru, Karnataka 570001",
        "latitude": 12.3134375,
        "longitude": 76.6498626,
        "specialties": ["General Medicine", "Surgery", "Emergency"],
        "phone": "+91 821 252 6200",
        "emergency": True
    },

    {
        "name": "Cheluvamba Hospital",
        "type": "Government",
        "address": "KR Hospital Compound, Irwin Road, Mysuru, Karnataka 570001",
        "latitude": 12.3141191,
        "longitude": 76.6493079,
        "specialties": ["Obstetrics", "Gynecology", "Pediatrics"],
        "phone": "+91 821 252 0512",
        "emergency": True
    },

    {
        "name": "District Hospital Mysore",
        "type": "Government",
        "address": "KRS Road, near Metagalli, Hebbal 1st Stage, Mysuru, Karnataka 570016",
        "latitude": 12.349812,
        "longitude": 76.6283509,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "+91 821 251 7555",
        "emergency": True
    },

    {
        "name": "ESI Hospital Mysuru",
        "type": "Government",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3383212,
        "longitude": 76.6323943,
        "specialties": ["General Medicine", "General Surgery", "Occupational Health"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Government Hospital, University of Mysore Campus",
        "type": "Government",
        "address": "Manasa Gangothiri, Mysuru, Karnataka 570006",
        "latitude": 12.3158,
        "longitude": 76.6089,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Government Ayurveda Medical College",
        "type": "Government",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3149518,
        "longitude": 76.6516333,
        "specialties": ["Ayurveda", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Railway Hospital Mysuru",
        "type": "Government",
        "address": "Mysuru Railway area, Mysuru, Karnataka",
        "latitude": 12.3257179,
        "longitude": 76.6356220,
        "specialties": ["General Medicine", "Occupational Health"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "General Hospital H.D. Kote",
        "type": "Government",
        "address": "H.D. Kote, Mysuru District, Karnataka",
        "latitude": 12.0134,
        "longitude": 76.3274,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "08228 255210",
        "emergency": True
    },

    {
        "name": "General Hospital Hunsur",
        "type": "Government",
        "address": "Hunsur, Mysuru District, Karnataka 571105",
        "latitude": 12.3035,
        "longitude": 76.2934,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "08222 252181",
        "emergency": True
    },

    {
        "name": "General Hospital K.R. Nagar",
        "type": "Government",
        "address": "K.R. Nagar, Mysuru District, Karnataka",
        "latitude": 12.5510,
        "longitude": 76.9020,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "08223 262205",
        "emergency": True
    },

    {
        "name": "General Hospital Periyapatna",
        "type": "Government",
        "address": "Periyapatna, Mysuru District, Karnataka",
        "latitude": 12.3347,
        "longitude": 76.1030,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Community Health Centre Jayanagar",
        "type": "Government",
        "address": "Jayanagar, Mysuru, Karnataka 570014",
        "latitude": 12.2947,
        "longitude": 76.6612,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "+91 821 246 1158",
        "emergency": False
    },

    {
        "name": "Community Health Centre Ashokapuram",
        "type": "Government",
        "address": "Ashokapuram, Mysuru, Karnataka 570008",
        "latitude": 12.2819,
        "longitude": 76.6512,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Community Health Centre Saligrama",
        "type": "Government",
        "address": "Saligrama, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.3814,
        "longitude": 76.8101,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "08223 283518",
        "emergency": False
    },

    {
        "name": "Community Health Centre Sargur",
        "type": "Government",
        "address": "Sargur, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9130,
        "longitude": 76.4640,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "08228 265240",
        "emergency": False
    },


    # ========================================================
    # PRIVATE MULTI-SPECIALITY HOSPITALS
    # ========================================================

    {
        "name": "Apollo BGS Hospitals",
        "type": "Private",
        "address": "Adichunchanagiri Road, Kuvempu Nagara, Mysuru, Karnataka 570023",
        "latitude": 12.2958525,
        "longitude": 76.6324329,
        "specialties": [
            "Cardiology", "Neurology", "Oncology",
            "Orthopedics", "General Medicine", "Emergency"
        ],
        "phone": "+91 8069049759",
        "emergency": True
    },

    {
        "name": "Narayana Hospital",
        "type": "Private",
        "address": "Devanur 2nd Stage, R.S. Naidu Nagar, Mysuru, Karnataka 570019",
        "latitude": 12.3452462,
        "longitude": 76.6730436,
        "specialties": ["Cardiology", "General Medicine", "Emergency"],
        "phone": "+91 8062154594",
        "emergency": True
    },

    {
        "name": "JSS Hospital",
        "type": "Private",
        "address": "Mahathma Gandhi Road, Mysuru, Karnataka 570004",
        "latitude": 12.2959818,
        "longitude": 76.6552185,
        "specialties": [
            "Cardiology", "Neurology", "Orthopedics",
            "General Medicine", "Pediatrics", "Emergency"
        ],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Manipal Hospital Mysore",
        "type": "Private",
        "address": "Bangalore-Mysore Ring Road Junction, Bannimantap, Mysuru 570015",
        "latitude": 12.3499281,
        "longitude": 76.6602337,
        "specialties": [
            "Cardiology", "Neurology", "Orthopedics",
            "General Medicine", "Emergency"
        ],
        "phone": "+91 1800 102 4647",
        "emergency": True
    },

    {
        "name": "Cauvery Heart and Multi-Speciality Hospital",
        "type": "Private",
        "address": "Malavalli-Mysore Road, Siddhartha Layout, Mysuru 570029",
        "latitude": 12.3053048,
        "longitude": 76.6959064,
        "specialties": ["Cardiology", "Gynecology", "General Medicine"],
        "phone": "+91 821 247 2424",
        "emergency": True
    },

    {
        "name": "DRM Multi Speciality Hospital",
        "type": "Private",
        "address": "Temple Road, Vontikoppal, Mysuru 570002",
        "latitude": 12.3248397,
        "longitude": 76.6320365,
        "specialties": [
            "Orthopedic", "Gastroenterology",
            "Pediatrics", "Emergency"
        ],
        "phone": "+91 81054 24247",
        "emergency": True
    },

    {
        "name": "Bhanavi Hospital",
        "type": "Private",
        "address": "Anikethana Road, Kuvempu Nagara, Mysuru 570023",
        "latitude": 12.2951,
        "longitude": 76.6310,
        "specialties": ["General Medicine", "Surgery", "Emergency"],
        "phone": "+91 821 664 4500",
        "emergency": True
    },

    {
        "name": "Vidyaranya Hospital",
        "type": "Private",
        "address": "2876, 4th Cross, N.S. Road, Chamundipuram, Mysuru 570004",
        "latitude": 12.2929,
        "longitude": 76.6552,
        "specialties": ["General Medicine", "Surgery"],
        "phone": "+91 821 233 0555",
        "emergency": True
    },

    {
        "name": "Gopala Gowda Shanthaveri Memorial Hospital",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3086153,
        "longitude": 76.6679217,
        "specialties": ["General Medicine", "Surgery", "Emergency"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Sigma Hospital",
        "type": "Private",
        "address": "Saraswathipuram, Mysuru, Karnataka",
        "latitude": 12.3003215,
        "longitude": 76.6239723,
        "specialties": [
            "General Surgery", "Cardiology",
            "Orthopedics", "ENT", "Urology"
        ],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "St. Joseph's Hospital",
        "type": "Private",
        "address": "Bannimantap, Mysuru, Karnataka 570015",
        "latitude": 12.3413224,
        "longitude": 76.6555307,
        "specialties": [
            "General Medicine", "Surgery",
            "Orthopedics", "ENT", "Gynecology"
        ],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "CSI Holdsworth Memorial Hospital",
        "type": "Private",
        "address": "Mandi Mohalla, Mysuru 570001",
        "latitude": 12.3204270,
        "longitude": 76.6500703,
        "specialties": ["General Medicine", "General Surgery"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Vikram Hospital",
        "type": "Private",
        "address": "Yadavagiri, Mysuru, Karnataka",
        "latitude": 12.32273,
        "longitude": 76.63734,
        "specialties": ["Cardiology", "Orthopedics", "General Medicine"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Nirmala Multispecialty Hospital",
        "type": "Private",
        "address": "Ring Road, Naranahalli, Alanahalli, Mysuru 570028",
        "latitude": 12.2906126,
        "longitude": 76.7050961,
        "specialties": ["Gynecology", "Cardiology"],
        "phone": "+91 821 401 4800",
        "emergency": True
    },

    {
        "name": "ClearMedi Multispeciality Hospital",
        "type": "Private",
        "address": "Hinkal Village, Kasaba Hobli, Mysuru 570017",
        "latitude": 12.3244583,
        "longitude": 76.6098097,
        "specialties": ["General Physician", "Emergency"],
        "phone": "+91 92170 20758",
        "emergency": True
    },

    {
        "name": "ClearMedi Radiant Hospital",
        "type": "Private",
        "address": "Vijay Nagar 3rd Stage, Mysuru 570030",
        "latitude": 12.3217974,
        "longitude": 76.6091748,
        "specialties": ["General Physician", "Oncology"],
        "phone": "+91 92170 20758",
        "emergency": True
    },

    {
        "name": "KVC Super Speciality Hospital",
        "type": "Private",
        "address": "Krishna Vilas Road, Subbarayanakere, Mysuru 570004",
        "latitude": 12.3079105,
        "longitude": 76.6454266,
        "specialties": ["Orthopedics", "Gynecology", "General Physician"],
        "phone": "+91 80500 78444",
        "emergency": True
    },

    {
        "name": "Supreme Multispeciality Hospital",
        "type": "Private",
        "address": "Kanakadasa Nagar, Dattagalli 3rd Stage, Mysuru 570033",
        "latitude": 12.2811315,
        "longitude": 76.6068077,
        "specialties": ["General Physician"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Nayana Kumar's Multi Speciality Hospital",
        "type": "Private",
        "address": "Dattagalli, Rajarajeshwari Nagar, Mysuru 570022",
        "latitude": 12.2824227,
        "longitude": 76.6047076,
        "specialties": ["ENT", "General Physician"],
        "phone": "+91 95133 10100",
        "emergency": True
    },

    {
        "name": "Sulakshaa Multi Speciality Hospital",
        "type": "Private",
        "address": "Vijay Nagar 3rd Stage, Mysuru 570017",
        "latitude": 12.3247949,
        "longitude": 76.6025505,
        "specialties": ["General Physician"],
        "phone": "+91 95357 47478",
        "emergency": True
    },

    {
        "name": "Alpha Hospital",
        "type": "Private",
        "address": "Subhash Nagar, Bannimantap, Mysuru 570015",
        "latitude": 12.3307,
        "longitude": 76.6387,
        "specialties": ["General Medicine", "Emergency"],
        "phone": "+91 821 200 1400",
        "emergency": True
    },

    {
        "name": "HCG Bharath Hospital & Institute of Oncology",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3490,
        "longitude": 76.6250,
        "specialties": ["Medical Oncology", "Surgical Oncology", "Radiation Oncology"],
        "phone": "Not listed",
        "emergency": True
    },


    # ========================================================
    # SPECIALITY HOSPITALS
    # ========================================================

    {
        "name": "JSS Dental Hospital",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.2964,
        "longitude": 76.6557,
        "specialties": ["Dental", "Oral Surgery"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Jayadeva Institute of Cardiovascular Sciences",
        "type": "Government",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3413058,
        "longitude": 76.6236452,
        "specialties": ["Cardiology", "Cardiac Surgery"],
        "phone": "Not listed",
        "emergency": True
    },

    {
        "name": "Vasan Eye Care Hospital",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.2961979,
        "longitude": 76.6436633,
        "specialties": ["Ophthalmology", "Eye Surgery"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Dr. Agarwal's Eye Hospital",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.2863447,
        "longitude": 76.6309373,
        "specialties": ["Ophthalmology", "Eye Surgery"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Bhagwan Mahaveer Darshan Eye Hospital",
        "type": "Private",
        "address": "Mysuru, Karnataka",
        "latitude": 12.3596533,
        "longitude": 76.6606623,
        "specialties": ["Ophthalmology"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # GOVERNMENT PRIMARY HEALTH CENTRES - MYSURU TALUK
    # ========================================================

    {
        "name": "AIWC (IMA) PHC",
        "type": "Government PHC",
        "address": "Mysuru Taluk, Mysuru, Karnataka",
        "latitude": 12.3060,
        "longitude": 76.6550,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Ashokapuram PHC",
        "type": "Government PHC",
        "address": "Ashokapuram, Mysuru, Karnataka",
        "latitude": 12.2819,
        "longitude": 76.6512,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Bannimantapa PHC",
        "type": "Government PHC",
        "address": "Bannimantapa, Mysuru, Karnataka",
        "latitude": 12.3305,
        "longitude": 76.6370,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Doora PHC",
        "type": "Government PHC",
        "address": "Doora, Mysuru Taluk, Mysuru",
        "latitude": 12.2800,
        "longitude": 76.6800,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Erangere PHC",
        "type": "Government PHC",
        "address": "Erangere, Mysuru Taluk, Mysuru",
        "latitude": 12.2805,
        "longitude": 76.6760,
        "specialties": ["Primary Care"],
        "phone": "0821 2447995",
        "emergency": False
    },

    {
        "name": "G.B. Palya PHC",
        "type": "Government PHC",
        "address": "G.B. Palya, Mysuru Taluk, Mysuru",
        "latitude": 12.2950,
        "longitude": 76.6850,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Jyothinagara PHC",
        "type": "Government PHC",
        "address": "Jyothinagara, Mysuru, Karnataka",
        "latitude": 12.3230,
        "longitude": 76.6220,
        "specialties": ["Primary Care"],
        "phone": "0821 2475834",
        "emergency": False
    },

    {
        "name": "Mellahalli PHC",
        "type": "Government PHC",
        "address": "Mellahalli, Mysuru Taluk, Mysuru",
        "latitude": 12.3500,
        "longitude": 76.5800,
        "specialties": ["Primary Care"],
        "phone": "0821 2593527",
        "emergency": False
    },

    {
        "name": "S.R. Hundi PHC",
        "type": "Government PHC",
        "address": "S.R. Hundi, Mysuru Taluk, Mysuru",
        "latitude": 12.2450,
        "longitude": 76.6900,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Sagarakatte PHC",
        "type": "Government PHC",
        "address": "Sagarakatte, Mysuru Taluk, Mysuru",
        "latitude": 12.4000,
        "longitude": 76.5400,
        "specialties": ["Primary Care"],
        "phone": "0821 2900841",
        "emergency": False
    },

    {
        "name": "Saraswathipuram PHC",
        "type": "Government PHC",
        "address": "Saraswathipuram, Mysuru, Karnataka",
        "latitude": 12.3067,
        "longitude": 76.6115,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Shanthinagara PHC",
        "type": "Government PHC",
        "address": "Shanthinagara, Mysuru Taluk, Mysuru",
        "latitude": 12.2980,
        "longitude": 76.6720,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "SMT PHC",
        "type": "Government PHC",
        "address": "Mysuru Taluk, Mysuru",
        "latitude": 12.3000,
        "longitude": 76.6500,
        "specialties": ["Primary Care"],
        "phone": "0821 2483587",
        "emergency": False
    },


    # ========================================================
    # HUNSUR TALUK HEALTH CENTRES
    # ========================================================

    {
        "name": "Bannikuppe PHC",
        "type": "Government PHC",
        "address": "Bannikuppe, Hunsur Taluk, Mysuru District",
        "latitude": 12.3400,
        "longitude": 76.2400,
        "specialties": ["Primary Care"],
        "phone": "08222 244328",
        "emergency": False
    },

    {
        "name": "Dharmapura PHC",
        "type": "Government PHC",
        "address": "Dharmapura, Hunsur Taluk, Mysuru District",
        "latitude": 12.2700,
        "longitude": 76.3100,
        "specialties": ["Primary Care"],
        "phone": "08222 245740",
        "emergency": False
    },

    {
        "name": "Doddhejjur PHC",
        "type": "Government PHC",
        "address": "Doddhejjur, Hunsur Taluk, Mysuru District",
        "latitude": 12.3000,
        "longitude": 76.2700,
        "specialties": ["Primary Care"],
        "phone": "08222 211353",
        "emergency": False
    },

    {
        "name": "Gavadagere PHC",
        "type": "Government PHC",
        "address": "Gavadagere, Hunsur Taluk, Mysuru District",
        "latitude": 12.2500,
        "longitude": 76.3300,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Hirekyatanahalli PHC",
        "type": "Government PHC",
        "address": "Hirekyatanahalli, Hunsur Taluk, Mysuru District",
        "latitude": 12.2200,
        "longitude": 76.2800,
        "specialties": ["Primary Care"],
        "phone": "08222 249151",
        "emergency": False
    },

    {
        "name": "Mulluru PHC",
        "type": "Government PHC",
        "address": "Mulluru, Hunsur Taluk, Mysuru District",
        "latitude": 12.2700,
        "longitude": 76.2500,
        "specialties": ["Primary Care"],
        "phone": "08222 248770",
        "emergency": False
    },

    {
        "name": "Rathnapuri PHC",
        "type": "Government PHC",
        "address": "Rathnapuri, Hunsur Taluk, Mysuru District",
        "latitude": 12.2800,
        "longitude": 76.2200,
        "specialties": ["Primary Care"],
        "phone": "08222 245626",
        "emergency": False
    },


    # ========================================================
    # H.D. KOTE TALUK
    # ========================================================

    {
        "name": "Annur PHC",
        "type": "Government PHC",
        "address": "Annur, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9800,
        "longitude": 76.2900,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Antharasanthe MTHU",
        "type": "Government Health Centre",
        "address": "Antharasanthe, H.D. Kote Taluk, Mysuru District",
        "latitude": 12.0100,
        "longitude": 76.1800,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "B Matakere PHC",
        "type": "Government PHC",
        "address": "B Matakere, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9700,
        "longitude": 76.2400,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Badagalapura PHC",
        "type": "Government PHC",
        "address": "Badagalapura, H.D. Kote Taluk, Mysuru District",
        "latitude": 12.0200,
        "longitude": 76.2800,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "DB Kuppe PHC",
        "type": "Government PHC",
        "address": "DB Kuppe, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9500,
        "longitude": 76.3500,
        "specialties": ["Primary Care"],
        "phone": "08228 210066",
        "emergency": False
    },

    {
        "name": "N Begur PHC",
        "type": "Government PHC",
        "address": "N Begur, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9000,
        "longitude": 76.3100,
        "specialties": ["Primary Care"],
        "phone": "08228 264425",
        "emergency": False
    },

    {
        "name": "Muluru PHC",
        "type": "Government PHC",
        "address": "Muluru, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9400,
        "longitude": 76.2800,
        "specialties": ["Primary Care"],
        "phone": "08228 210033",
        "emergency": False
    },

    {
        "name": "Sagare PHC",
        "type": "Government PHC",
        "address": "Sagare, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9300,
        "longitude": 76.4100,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Shanthipura PHC",
        "type": "Government PHC",
        "address": "Shanthipura, H.D. Kote Taluk, Mysuru District",
        "latitude": 11.9700,
        "longitude": 76.3900,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # NANJANGUD TALUK
    # ========================================================

    {
        "name": "Dasanur PHC",
        "type": "Government PHC",
        "address": "Dasanur, Nanjangud Taluk, Mysuru District",
        "latitude": 12.0600,
        "longitude": 76.7200,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Devanur PHC",
        "type": "Government PHC",
        "address": "Devanur, Nanjangud Taluk, Mysuru District",
        "latitude": 12.0500,
        "longitude": 76.6900,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Hura PHC",
        "type": "Government PHC",
        "address": "Hura, Nanjangud Taluk, Mysuru District",
        "latitude": 12.0800,
        "longitude": 76.6900,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Eregowdanahundi Maternity Hospital",
        "type": "Government",
        "address": "Eregowdanahundi, Nanjangud Taluk, Mysuru District",
        "latitude": 12.0900,
        "longitude": 76.6500,
        "specialties": ["Maternity", "Gynecology", "Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # K.R. NAGAR TALUK
    # ========================================================

    {
        "name": "Adaguru PHC",
        "type": "Government PHC",
        "address": "Adaguru, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.5700,
        "longitude": 76.8500,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Gandanahalli PHC",
        "type": "Government PHC",
        "address": "Gandanahalli, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.5200,
        "longitude": 76.8200,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Malali PHC",
        "type": "Government PHC",
        "address": "Malali, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.5000,
        "longitude": 76.8500,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Meluru PHC",
        "type": "Government PHC",
        "address": "Meluru, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.5400,
        "longitude": 76.8800,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Mirle PHC",
        "type": "Government PHC",
        "address": "Mirle, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.4700,
        "longitude": 76.8500,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Munduru PHC",
        "type": "Government PHC",
        "address": "Munduru, K.R. Nagar Taluk, Mysuru District",
        "latitude": 12.5200,
        "longitude": 76.9000,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # T. NARASIPURA TALUK
    # ========================================================

    {
        "name": "Doddamulagudu PHC",
        "type": "Government PHC",
        "address": "T. Narasipura Taluk, Mysuru District",
        "latitude": 12.1900,
        "longitude": 76.9000,
        "specialties": ["Primary Care"],
        "phone": "82312 40771",
        "emergency": False
    },

    {
        "name": "Gargeshwari PHC",
        "type": "Government PHC",
        "address": "Gargeshwari, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.1900,
        "longitude": 76.8500,
        "specialties": ["Primary Care"],
        "phone": "82272 62235",
        "emergency": False
    },

    {
        "name": "Malangi PHC",
        "type": "Government PHC",
        "address": "Malangi, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.1800,
        "longitude": 76.8700,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Muguru PHC",
        "type": "Government PHC",
        "address": "Muguru, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.2400,
        "longitude": 76.9000,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Rangasamudra PHC",
        "type": "Government PHC",
        "address": "Rangasamudra, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.2100,
        "longitude": 76.9200,
        "specialties": ["Primary Care"],
        "phone": "82272 75067",
        "emergency": False
    },

    {
        "name": "Somanathapura PHC",
        "type": "Government PHC",
        "address": "Somanathapura, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.2700,
        "longitude": 76.9000,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Sosale PHC",
        "type": "Government PHC",
        "address": "Sosale, T. Narasipura Taluk, Mysuru District",
        "latitude": 12.2500,
        "longitude": 76.9500,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # PERIYAPATNA TALUK
    # ========================================================

    {
        "name": "Attigodu MTHU",
        "type": "Government Health Centre",
        "address": "Attigodu, Periyapatna Taluk, Mysuru District",
        "latitude": 12.3300,
        "longitude": 76.0500,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Doddabelalu PHC",
        "type": "Government PHC",
        "address": "Doddabelalu, Periyapatna Taluk, Mysuru District",
        "latitude": 12.3500,
        "longitude": 76.0200,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Ravandooru PHC",
        "type": "Government PHC",
        "address": "Ravandooru, Periyapatna Taluk, Mysuru District",
        "latitude": 12.3600,
        "longitude": 76.1300,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Sangarashettahalli PHC",
        "type": "Government PHC",
        "address": "Sangarashettahalli, Periyapatna Taluk, Mysuru District",
        "latitude": 12.3900,
        "longitude": 76.0800,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Shyanubhoganahalli MTHU",
        "type": "Government Health Centre",
        "address": "Shyanubhoganahalli, Periyapatna Taluk, Mysuru District",
        "latitude": 12.4000,
        "longitude": 76.1200,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },


    # ========================================================
    # ADDITIONAL MYSURU PRIVATE FACILITIES
    # ========================================================

    {
        "name": "K.R. Hospital Road Private Medical Centre",
        "type": "Private",
        "address": "Devaraja Mohalla, Mysuru, Karnataka",
        "latitude": 12.3145,
        "longitude": 76.6510,
        "specialties": ["General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "City Health Centre Mysuru",
        "type": "Private",
        "address": "Kalidasa Road, Vijayanagar/Vilas Mohalla area, Mysuru",
        "latitude": 12.3235,
        "longitude": 76.6305,
        "specialties": ["General Medicine", "Primary Care"],
        "phone": "+91 821 2410047",
        "emergency": False
    },

    {
        "name": "Urban Primary Health Centre HHMBG C1",
        "type": "Government UPHC",
        "address": "KC Layout, Mysuru, Karnataka",
        "latitude": 12.2880,
        "longitude": 76.6350,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "+91 821 2477456",
        "emergency": False
    },

    {
        "name": "Government Primary Health Centre Kuvempunagara",
        "type": "Government PHC",
        "address": "Kuvempunagara North, TK Layout, Mysuru 570009",
        "latitude": 12.2910,
        "longitude": 76.6200,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Primary Health Centre Saraswathipuram",
        "type": "Government PHC",
        "address": "2nd Cross Road, Saraswathipuram, Mysuru 570009",
        "latitude": 12.3050,
        "longitude": 76.6110,
        "specialties": ["Primary Care"],
        "phone": "Not listed",
        "emergency": False
    },

    {
        "name": "Government Health Centre N.R. Mohalla",
        "type": "Government",
        "address": "N.R. Mohalla, Gayathri Puram, Mysuru 570007",
        "latitude": 12.3310,
        "longitude": 76.6550,
        "specialties": ["Primary Care", "General Medicine"],
        "phone": "Not listed",
        "emergency": False
    },

]


# ============================================================
# DOCTOR-TITLE  -->  HOSPITAL SPECIALTY MATCHING
# ============================================================
# Maps each individual doctor title that appears in DOCTOR_MAP above
# (e.g. "Cardiologist", "Hepatologist / Gastroenterologist") to the
# specialty keywords used inside HOSPITAL_LIST's "specialties" field.
#
# Compound titles like "Hepatologist / Gastroenterologist" or
# "Dermatologist / General Physician" are split on "/" and each part
# is looked up individually, then the keyword lists are combined.
# If nothing matches, we fall back to General Physician / General
# Medicine hospitals so results are never empty.
# ============================================================

SINGLE_DOCTOR_TITLE_KEYWORDS = {
    "dermatologist": [],                       # no dedicated skin hospital in list -> falls back below
    "allergist": [],                           # falls back below
    "immunologist": [],                        # falls back below
    "gastroenterologist": ["gastroenterology"],
    "hepatologist": ["gastroenterology"],       # closest available specialty
    "infectious disease specialist": [],        # falls back below
    "endocrinologist": [],                     # falls back below
    "pulmonologist": [],                       # falls back below
    "cardiologist": ["cardiology", "cardiac"],
    "neurologist": ["neurology", "neuro"],
    "orthopedician": ["orthopedic", "orthopedics", "ortho"],
    "orthopedic surgeon": ["orthopedic", "orthopedics", "ortho"],
    "general physician": ["general physician", "general medicine"],
    "general surgeon": ["general surgery", "surgery"],
    "proctologist": ["general surgery", "surgery"],
    "vascular surgeon": ["general surgery", "surgery"],
    "rheumatologist": ["orthopedic", "orthopedics"],
    "ent specialist": ["ent"],
    "otolaryngologist": ["ent"],
    "urologist": ["urology"],
    "oncologist": ["oncology", "medical oncology", "surgical oncology", "radiation oncology"],
    "ophthalmologist": ["ophthalmology", "eye"],
    "dentist": ["dental", "oral surgery"],
    "pediatrician": ["pediatrics"],
    "gynecologist": ["gynecology", "obstetrics", "maternity"],
    "obstetrician": ["gynecology", "obstetrics", "maternity"],
}

FALLBACK_SPECIALTY_KEYWORDS = ["general physician", "general medicine"]


def get_specialty_keywords_for_doctor(doctor_title):
    """
    Given a doctor title string from DOCTOR_MAP (e.g. 'Cardiologist' or
    'Hepatologist / Gastroenterologist'), return the combined list of
    hospital-specialty keywords to match against.

    Handles "X / Y" compound titles by splitting and looking up each part.
    Falls back to General Physician / General Medicine keywords if no
    part of the title has a specific mapping.
    """
    if not doctor_title:
        return []

    parts = [p.strip().lower() for p in doctor_title.split("/")]
    combined_keywords = []

    for part in parts:
        keywords = SINGLE_DOCTOR_TITLE_KEYWORDS.get(part)
        if keywords:
            for kw in keywords:
                if kw not in combined_keywords:
                    combined_keywords.append(kw)

    if not combined_keywords:
        combined_keywords = FALLBACK_SPECIALTY_KEYWORDS

    return combined_keywords


def hospital_matches_specialty(hospital, specialty_keywords):
    """
    Returns True if any of the hospital's specialties contains any of the
    given keywords (case-insensitive, substring match).
    """
    if not specialty_keywords:
        return False
    hospital_specialties_lower = [s.lower() for s in hospital.get("specialties", [])]
    for keyword in specialty_keywords:
        for specialty in hospital_specialties_lower:
            if keyword in specialty:
                return True
    return False


# ============================================================
# HAVERSINE DISTANCE CALCULATION
# ============================================================

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on Earth (specified in decimal degrees) in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# ============================================================
# /hospitals ENDPOINT
# ============================================================
# Example calls from Android:
#   GET /hospitals?lat=12.2958&lon=76.6394
#   GET /hospitals?lat=12.2958&lon=76.6394&doctor=Cardiologist
#   GET /hospitals?lat=12.2958&lon=76.6394&doctor=Hepatologist%20/%20Gastroenterologist
#   GET /hospitals?lat=12.2958&lon=76.6394&doctor=Cardiologist&emergency_only=true
#
# Response: JSON list of hospitals sorted so that:
#   1. Hospitals whose specialty matches the recommended doctor come first
#   2. Within each group, hospitals are sorted by distance (nearest first)
# ============================================================

@app.route('/hospitals', methods=['GET'])
def get_nearby_hospitals():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)

        if lat is None or lon is None:
            return jsonify({
                "error": "Missing required parameters 'lat' and 'lon'. "
                         "Example: /hospitals?lat=12.2958&lon=76.6394"
            }), 400

        # Optional: filter/prioritize by recommended doctor's specialty
        # Pass the exact string from a /predict response's "recommended_doctor" field.
        doctor_title = request.args.get('doctor', default=None, type=str)
        specialty_keywords = get_specialty_keywords_for_doctor(doctor_title)

        # Optional: only return hospitals with emergency services
        emergency_only = request.args.get('emergency_only', default='false', type=str).lower() == 'true'

        # Optional: limit number of results returned (default 20)
        limit = request.args.get('limit', default=20, type=int)

        results = []
        for hospital in HOSPITAL_LIST:
            if emergency_only and not hospital.get("emergency", False):
                continue

            distance_km = haversine_distance_km(lat, lon, hospital["latitude"], hospital["longitude"])
            is_match = hospital_matches_specialty(hospital, specialty_keywords)

            results.append({
                "name": hospital["name"],
                "type": hospital["type"],
                "address": hospital["address"],
                "latitude": hospital["latitude"],
                "longitude": hospital["longitude"],
                "specialties": hospital["specialties"],
                "phone": hospital["phone"],
                "emergency": hospital["emergency"],
                "distance_km": round(distance_km, 2),
                "specialty_match": is_match
            })

        # Sort: specialty matches first (True sorts before False using "not is_match"),
        # then by distance ascending within each group.
        results.sort(key=lambda h: (not h["specialty_match"], h["distance_km"]))

        return jsonify({
            "count": len(results[:limit]),
            "doctor_filter_used": doctor_title,
            "hospitals": results[:limit]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
