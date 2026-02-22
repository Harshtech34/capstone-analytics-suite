import requests
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

# Load env
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_GEOCODE_API")
logger = logging.getLogger("capstone")

def geocode_address(address: str):
    """
    Convert address string to latitude and longitude using Google Maps API.
    """
    if not GOOGLE_API_KEY:
        logger.error("Google API key not found in .env")
        return None, None

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if data["status"] == "OK":
            loc = data["results"][0]["geometry"]["location"]
            logger.info(f"Geocoded address: {address} -> {loc['lat']}, {loc['lng']}")
            return loc["lat"], loc["lng"]
        else:
            logger.warning(f"Google Maps API failed: {data['status']}")
            return None, None
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        return None, None
