"""
predict.py
Reusable inference module. Import `predict_disease()` from here in your
Flask API (app.py) later.
"""

import pickle
import json
import numpy as np
import pandas as pd

MODEL_PATH = "model/random_forest.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
SYMPTOMS_PATH = "model/symptom_list.json"

# Load once at import time
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SYMPTOMS_PATH, "r") as f:
    SYMPTOM_LIST = json.load(f)

SYMPTOM_SET = set(SYMPTOM_LIST)


def predict_disease(symptoms: list[str], top_n: int = 3):
    """
    symptoms: list of symptom strings, e.g. ["itching", "skin_rash", "fatigue"]
              (must match the exact column names in symptom_list.json —
              the Flask layer will handle fuzzy matching/normalization later)

    Returns: {
        "prediction": "Fungal infection",
        "confidence": 0.87,
        "top_matches": [
            {"disease": "Fungal infection", "confidence": 0.87},
            {"disease": "Allergy", "confidence": 0.08},
            {"disease": "Acne", "confidence": 0.03}
        ],
        "unrecognized_symptoms": []   # symptoms not found in the trained vocabulary
    }
    """
    unrecognized = [s for s in symptoms if s not in SYMPTOM_SET]
    valid_symptoms = [s for s in symptoms if s in SYMPTOM_SET]

    # Build the input vector (1 x 132), 1 if symptom present, else 0
    input_vector = np.zeros(len(SYMPTOM_LIST))
    for s in valid_symptoms:
        idx = SYMPTOM_LIST.index(s)
        input_vector[idx] = 1

    # Wrap as a DataFrame with the original column names so sklearn doesn't
    # warn about missing feature names (and so column order is unambiguous)
    input_df = pd.DataFrame([input_vector], columns=SYMPTOM_LIST)

    # Predict probabilities across all 41 diseases
    probabilities = model.predict_proba(input_df)[0]

    # Get top_n predictions sorted by confidence
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_matches = [
        {
            "disease": label_encoder.inverse_transform([idx])[0],
            "confidence": round(float(probabilities[idx]), 4)
        }
        for idx in top_indices
    ]

    return {
        "prediction": top_matches[0]["disease"],
        "confidence": top_matches[0]["confidence"],
        "top_matches": top_matches,
        "unrecognized_symptoms": unrecognized
    }


if __name__ == "__main__":
    # Quick manual test
    test_symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
    result = predict_disease(test_symptoms)
    print(json.dumps(result, indent=2))

    print("\n--- second test with unrecognized symptom ---")
    test_symptoms_2 = ["chest_pain", "sweating", "made_up_symptom"]
    result2 = predict_disease(test_symptoms_2)
    print(json.dumps(result2, indent=2))
