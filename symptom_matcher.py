"""
symptom_matcher.py
Converts messy, free-text symptom input (typed OR from voice-to-text) into
the exact symptom keys the model was trained on.

Handles:
  - "stomach ache" -> "stomach_pain"
  - spaces/hyphens instead of underscores
  - minor typos ("hedache" -> "headache")
  - comma or "and"-separated lists ("fever, cough and headache")
"""

import json
import re
from thefuzz import process, fuzz

with open("model/symptom_list.json", "r") as f:
    RAW_SYMPTOM_LIST = json.load(f)

# Clean up any stray whitespace in the original dataset's column names
# (e.g. "spotting_ urination" -> "spotting_urination") while keeping a
# lookup back to the ORIGINAL name the model was actually trained on.
CLEAN_TO_ORIGINAL = {}
for s in RAW_SYMPTOM_LIST:
    clean = re.sub(r"\s+", "_", s.strip())
    clean = re.sub(r"_+", "_", clean)
    CLEAN_TO_ORIGINAL[clean] = s

CLEAN_SYMPTOMS = list(CLEAN_TO_ORIGINAL.keys())

# Human-readable versions for fuzzy matching against natural speech,
# e.g. "stomach_pain" -> "stomach pain"
READABLE_TO_CLEAN = {c.replace("_", " "): c for c in CLEAN_SYMPTOMS}
READABLE_LIST = list(READABLE_TO_CLEAN.keys())

# Common synonyms people actually say out loud, mapped to the closest
# trained symptom phrase. Extend this dict over time as you test with
# real voice input.
SYNONYMS = {
    "stomach ache": "stomach pain",
    "stomachache": "stomach pain",
    "tummy ache": "stomach pain",
    "throwing up": "vomiting",
    "runny nose": "continuous sneezing",
    "high temperature": "high fever",
    "temperature": "mild fever",
    "fever": "mild fever",              # NEW: plain "fever" now maps to mild_fever
    "feaver": "mild fever",             # NEW: common typo/mishearing
    "cant breathe": "breathlessness",
    "can't breathe": "breathlessness",
    "shortness of breath": "breathlessness",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhausted": "fatigue",
    "body ache": "muscle pain",
    "body pain": "muscle pain",
    "loose motion": "diarrhoea",
    "loose motions": "diarrhoea",
    "yellow eyes": "yellowing of eyes",
    "yellow skin": "yellowish skin",
    "dizzy": "dizziness",
    "headache": "headache",
    "head ache": "headache",            # NEW: two-word phrasing now matches
    "sore throat": "throat irritation",
}

FUZZY_THRESHOLD = 78  # 0-100, tuned to avoid wrong matches on short words


def _split_input(text: str) -> list[str]:
    """Break 'fever, cough and headache' into ['fever', 'cough', 'headache']"""
    text = text.lower().strip()
    text = re.sub(r"\band\b", ",", text)
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]


def normalize_symptoms(raw_text_or_list) -> dict:
    """
    Accepts either a comma-separated string (typical from voice-to-text)
    or a list of strings (typical from checkbox UI).

    Returns: {
        "matched": ["stomach_pain", "vomiting"],       # exact model keys, ready for predict_disease()
        "matched_readable": [("stomach ache", "stomach_pain"), ...],  # what was heard -> what it mapped to
        "unmatched": ["some gibberish"]                 # couldn't confidently match, show user for confirmation
    }
    """
    if isinstance(raw_text_or_list, str):
        phrases = _split_input(raw_text_or_list)
    else:
        phrases = [p.lower().strip() for p in raw_text_or_list if p.strip()]

    matched = []
    matched_readable = []
    unmatched = []

    for phrase in phrases:
        phrase_lower = phrase.lower().strip()

        # 0. Direct exact-key match first (handles the case where the caller
        # already sends valid trained symptom keys like "skin_rash")
        underscored = re.sub(r"[\s-]+", "_", phrase_lower)
        underscored = re.sub(r"_+", "_", underscored).strip("_")
        if underscored in CLEAN_TO_ORIGINAL:
            matched.append(CLEAN_TO_ORIGINAL[underscored])
            matched_readable.append((phrase, underscored))
            continue

        phrase_clean = re.sub(r"[^a-z\s]", "", phrase_lower).strip()
        if not phrase_clean:
            continue

        # 1. Direct synonym hit
        candidate_readable = SYNONYMS.get(phrase_clean, phrase_clean)

        # 2. Exact readable match
        if candidate_readable in READABLE_TO_CLEAN:
            clean_key = READABLE_TO_CLEAN[candidate_readable]
            matched.append(CLEAN_TO_ORIGINAL[clean_key])
            matched_readable.append((phrase, clean_key))
            continue

        # 3. Fuzzy match against the full readable symptom list
        best_match, score = process.extractOne(
            candidate_readable, READABLE_LIST, scorer=fuzz.token_sort_ratio
        )
        if score >= FUZZY_THRESHOLD:
            clean_key = READABLE_TO_CLEAN[best_match]
            matched.append(CLEAN_TO_ORIGINAL[clean_key])
            matched_readable.append((phrase, clean_key))
        else:
            unmatched.append(phrase)

    return {
        "matched": list(dict.fromkeys(matched)),  # dedupe, preserve order
        "matched_readable": matched_readable,
        "unmatched": unmatched
    }


if __name__ == "__main__":
    tests = [
        "stomach ache, throwing up and tired",
        "hedache, feaver, cofh",
        "yellow eyes, dark urine",
        "fever, head ache",
    ]
    for t in tests:
        print(f"\nInput: {t}")
        print(json.dumps(normalize_symptoms(t), indent=2))
