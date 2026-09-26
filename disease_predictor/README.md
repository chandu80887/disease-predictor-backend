# AI-Powered Symptom-Based Disease Prediction System — Backend

## Setup
```
pip install -r requirements.txt
python3 train_model.py   # already trained, but rerun if you retrain
python3 app.py            # starts Flask server on http://0.0.0.0:5000
```

## Test without running a live server
```
python3 test_api.py
```

## Endpoints
- GET  /                      health check
- GET  /symptoms               list of 132 valid symptoms (human-readable, for Android UI)
- POST /predict                 body: {"text": "stomach ache, tired"} OR {"symptoms": ["itching","skin_rash"]}
- GET  /disease/<name>/info     description, precautions, medications, diet, workout

## Files
- data/            raw dataset (Training.csv + recommendation tables)
- model/           trained Random Forest, label encoder, symptom list, metrics
- train_model.py   retrain the model
- predict.py       core prediction logic (loads model, returns top-3 with confidence)
- symptom_matcher.py  fuzzy-matches free text/voice input to trained symptom keys
- app.py           Flask API tying it all together
- test_api.py       exercises every endpoint

## Next steps for the Android app (Kotlin)
Point Retrofit/OkHttp at these endpoints. When you deploy (Render/Railway/etc. — 
free tier is fine for a college project), swap localhost for the deployed URL.
