import requests
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from backend.satellite_query import get_access_token

load_dotenv()

DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

def download_scene(scene_id: str, scene_name: str, save_dir: str = "data/scenes"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(save_dir, f"{scene_name}.zip")

    if os.path.exists(output_path):
        print(f"Scene already downloaded: {scene_name}")
        return output_path

    print(f"Downloading {scene_name}...")
    token = get_access_token()

    url = f"{DOWNLOAD_URL}({scene_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%", end="")
    print(f"\nSaved to {output_path}")
    return output_path

def extract_bands(zip_path: str, output_dir: str = "data/scenes"):
    print(f"Extracting bands from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        tiff_files = [f for f in z.namelist() if f.endswith(".tiff") or f.endswith(".img")]
        vv_files = [f for f in tiff_files if "vv" in f.lower() or "VV" in f]
        vh_files = [f for f in tiff_files if "vh" in f.lower() or "VH" in f]

        extracted = []
        for band_file in vv_files[:1] + vh_files[:1]:
            dest = os.path.join(output_dir, os.path.basename(band_file))
            with z.open(band_file) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted.append(dest)
            print(f"  Extracted: {os.path.basename(band_file)}")

    return extracted

if __name__ == "__main__":
    from backend.satellite_query import search_scenes
    scenes = search_scenes("2024-01-01", "2024-02-01")
    if scenes:
        first = scenes[0]
        print(f"Downloading first scene: {first['name']}")
        zip_path = download_scene(first["id"], first["name"])
        bands = extract_bands(zip_path)
        print(f"\nExtracted bands: {bands}")
    else:
        print("No scenes found to download.")
