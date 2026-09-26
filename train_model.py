"""
train_model.py
Trains a Random Forest Classifier to predict disease from 132 symptoms.
Saves the trained model + label encoder + symptom list for use in the Flask API.
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------- 1. Load data ----------
df = pd.read_csv("data/Training.csv")

# Drop any stray "Unnamed" columns some copies of this dataset have
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

symptom_columns = [c for c in df.columns if c != "prognosis"]
X = df[symptom_columns]
y_raw = df["prognosis"].str.strip()  # dataset has trailing spaces on some labels e.g. "Diabetes "

# ---------- 2. Encode disease labels ----------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"Samples: {X.shape[0]}, Symptoms: {X.shape[1]}, Diseases: {len(label_encoder.classes_)}")

# ---------- 3. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. Train Random Forest ----------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

print("\nClassification Report (first 5 classes shown for brevity):")
report = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)
for cls in list(label_encoder.classes_)[:5]:
    r = report[cls]
    print(f"  {cls}: precision={r['precision']:.2f} recall={r['recall']:.2f} f1={r['f1-score']:.2f}")

# ---------- 6. Feature importance (top 15 symptoms overall) ----------
importances = pd.Series(model.feature_importances_, index=symptom_columns)
top_features = importances.sort_values(ascending=False).head(15)
print("\nTop 15 most important symptoms:")
for sym, score in top_features.items():
    print(f"  {sym}: {score:.4f}")

# ---------- 7. Save artifacts ----------
with open("model/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("model/symptom_list.json", "w") as f:
    json.dump(symptom_columns, f, indent=2)

with open("model/metrics.json", "w") as f:
    json.dump({
        "test_accuracy": acc,
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "n_samples": int(X.shape[0]),
        "n_symptoms": int(X.shape[1]),
        "n_diseases": int(len(label_encoder.classes_)),
    }, f, indent=2)

print("\nSaved: model/random_forest.pkl, model/label_encoder.pkl, model/symptom_list.json, model/metrics.json")
