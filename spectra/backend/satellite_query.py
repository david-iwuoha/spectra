import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

COPERNICUS_USER = os.getenv("COPERNICUS_USER")
COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")

AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"

def get_access_token():
    response = requests.post(AUTH_URL, data={
        "grant_type": "password",
        "username": COPERNICUS_USER,
        "password": COPERNICUS_PASSWORD,
        "client_id": "cdse-public",
    })
    response.raise_for_status()
    return response.json()["access_token"]

def search_scenes(date_from: str, date_to: str):
    # bbox = [min_lon, min_lat, max_lon, max_lat] — Niger Delta
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": [5.0, 4.0, 8.5, 6.5],
        "datetime": f"{date_from}T00:00:00Z/{date_to}T00:00:00Z",
        "limit": 5
    }

    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(STAC_URL, json=payload, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Content-Length: {len(response.content)}")

    if not response.content:
        print("Empty response from server.")
        return []

    try:
        data = response.json()
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw: {response.text[:300]}")
        return []

    features = data.get("features", [])
    scenes = []
    for item in features:
        props = item.get("properties", {})
        scenes.append({
            "id": item["id"],
            "name": item["id"],
            "date": props.get("datetime", ""),
            "platform": props.get("platform", ""),
            "assets": list(item.get("assets", {}).keys())
        })

    return scenes

if __name__ == "__main__":
    print("Authenticating...")
    token = get_access_token()
    print("Token obtained.")

    print("Searching for Sentinel-1 scenes over Niger Delta...")
    scenes = search_scenes("2024-01-01", "2024-03-01")

    if scenes:
        print(f"\nFound {len(scenes)} scenes:")
        for s in scenes:
            print(f"  - {s['name']}")
            print(f"    Date: {s['date'][:10]}")
            print(f"    Assets: {s['assets']}")
    else:
        print("No scenes found.")
