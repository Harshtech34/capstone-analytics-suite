# src/google_places.py
import os, requests, logging
from urllib.parse import urlencode
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.getenv("GOOGLE_API_KEY")
logger = logging.getLogger("capstone.places")

AUTOCOMPLETE_URL = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

def autocomplete_places(input_text, session_token=None, country=None):
    if not API_KEY or not input_text:
        return []
    params = {"input": input_text, "key": API_KEY, "types": "address", "components": f"country:{country}" if country else None}
    # remove None values
    params = {k: v for k, v in params.items() if v}
    r = requests.get(AUTOCOMPLETE_URL, params=params, timeout=10)
    j = r.json()
    if j.get("status") != "OK":
        logger.warning("Places autocomplete status: %s", j.get("status"))
        return []
    suggestions = [{"description": p["description"], "place_id": p["place_id"]} for p in j["predictions"]]
    return suggestions

def place_details(place_id):
    if not API_KEY or not place_id:
        return None
    params = {"place_id": place_id, "key": API_KEY, "fields": "formatted_address,geometry"}
    r = requests.get(DETAILS_URL, params=params, timeout=10)
    j = r.json()
    if j.get("status") != "OK":
        logger.warning("Place details status: %s", j.get("status"))
        return None
    res = j["result"]
    loc = res.get("geometry", {}).get("location")
    return {
        "address": res.get("formatted_address"),
        "lat": loc.get("lat") if loc else None,
        "lon": loc.get("lng") if loc else None
    }
