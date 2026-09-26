import json
from app import app

client = app.test_client()

print("=== GET / ===")
r = client.get("/")
print(r.status_code, json.dumps(r.get_json(), indent=2))

print("\n=== GET /symptoms (first 5) ===")
r = client.get("/symptoms")
data = r.get_json()
print(r.status_code, "count:", data["count"], "sample:", data["symptoms"][:5])

print("\n=== POST /predict with 'text' (voice-style input) ===")
r = client.post("/predict", json={"text": "stomach ache, throwing up, tired, dark urine"})
print(r.status_code, json.dumps(r.get_json(), indent=2))

print("\n=== POST /predict with exact 'symptoms' list ===")
r = client.post("/predict", json={"symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]})
print(r.status_code, json.dumps(r.get_json(), indent=2))

print("\n=== POST /predict with garbage input ===")
r = client.post("/predict", json={"text": "asdkjaksjd, qweqweqwe"})
print(r.status_code, json.dumps(r.get_json(), indent=2))

print("\n=== GET /disease/Fungal infection/info ===")
r = client.get("/disease/Fungal infection/info")
print(r.status_code, json.dumps(r.get_json(), indent=2))
