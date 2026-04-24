import numpy as np
import torch
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
from scipy.ndimage import uniform_filter
from shapely.geometry import shape, mapping
from pathlib import Path
import segmentation_models_pytorch as smp
import json
import logging

# ── Phase C: Look-alike Classifier ──────────────────────────────────────────
from backend.lookalike_classifier import LookalikeClassifier
from backend.wind_context import WindContextLayer, drift_arrow_geojson


logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PATCH_SIZE = 256
DEVICE = "cpu"

_lookalike_classifier = LookalikeClassifier(
    model_path=MODELS_DIR / "lookalike_model.pth"
)
_wind_context = WindContextLayer()

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )
    model.load_state_dict(torch.load(
        MODELS_DIR / "spectra_model.pth",
        map_location=DEVICE
    ))
    model.eval()
    return model

def to_db(array):
    array = np.where(array > 0, array, 1e-10)
    return 10 * np.log10(array)

def lee_filter(array, size=7):
    mean = uniform_filter(array.astype(float), size)
    mean_sq = uniform_filter(array.astype(float) ** 2, size)
    variance = mean_sq - mean ** 2
    noise_var = np.mean(variance)
    weights = variance / (variance + noise_var + 1e-10)
    return mean + weights * (array - mean)

def normalize(array):
    mn, mx = array.min(), array.max()
    if mx - mn == 0:
        return np.zeros_like(array)
    return (array - mn) / (mx - mn)

def preprocess_patch(vv_arr, vh_arr):
    vv = normalize(lee_filter(to_db(vv_arr.astype(np.float32))))
    vh = normalize(lee_filter(to_db(vh_arr.astype(np.float32))))
    patch = np.stack([vv, vh], axis=0)
    return torch.tensor(patch).unsqueeze(0).float()

def run_detection(vv_path: str, vh_path: str = None):

    model = load_model()
    print(f"Model loaded. Running detection on {Path(vv_path).name}...")

    with rasterio.open(vv_path) as vv_src:
        height = vv_src.height
        width = vv_src.width
        transform = vv_src.transform
        crs = vv_src.crs

        vh_src = rasterio.open(vh_path) if vh_path else None

        prob_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)

        best_patch_vv = None
        best_patch_conf = -1.0

        CHUNK = 1024
        patch_count = 0

        for row_off in range(0, height - CHUNK + 1, CHUNK):
            for col_off in range(0, width - CHUNK + 1, CHUNK):
                from rasterio.windows import Window
                window = Window(col_off, row_off, CHUNK, CHUNK)

                vv_chunk = vv_src.read(1, window=window).astype(np.float32)
                vh_chunk = vh_src.read(1, window=window).astype(np.float32) if vh_src else vv_chunk.copy()

                ch, cw = vv_chunk.shape
                for i in range(0, ch - PATCH_SIZE + 1, PATCH_SIZE):
                    for j in range(0, cw - PATCH_SIZE + 1, PATCH_SIZE):
                        vv_tile = vv_chunk[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        vh_tile = vh_chunk[i:i+PATCH_SIZE, j:j+PATCH_SIZE]

                        inp = preprocess_patch(vv_tile, vh_tile)
                        with torch.no_grad():
                            pred = torch.sigmoid(model(inp))

                        prob = pred.squeeze().numpy()
                        r0, c0 = row_off + i, col_off + j
                        prob_map[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE] += prob
                        count_map[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE] += 1

                        patch_conf = float(prob.mean())
                        if patch_conf > best_patch_conf:
                            best_patch_conf = patch_conf
                            best_patch_vv = vv_tile.copy()

                        patch_count += 1

                print(f"  Chunk ({row_off},{col_off}) done — {patch_count} patches")

                if patch_count >= 16:
                    break
            if patch_count >= 16:
                break

        if vh_src:
            vh_src.close()

    count_map = np.where(count_map == 0, 1, count_map)
    prob_map = prob_map / count_map

    binary_mask = (prob_map > 0.5).astype(np.uint8)

    spill_pixels = binary_mask.sum()
    confidence = float(prob_map[binary_mask == 1].mean()) if spill_pixels > 0 else 0.0

    pixel_area_m2 = 10 * 10
    area_km2 = round((spill_pixels * pixel_area_m2) / 1e6, 4)

    polygons = []
    if spill_pixels > 0:
        for geom, val in shapes(binary_mask, transform=transform):
            if val == 1:
                polygons.append(shape(geom))

    if polygons:
        from shapely.ops import unary_union
        merged = unary_union(polygons)
        geojson_polygon = mapping(merged)
    else:
        geojson_polygon = None

    # ── Phase C: Look-alike classifier ──────────────────────────────────────
    if spill_pixels > 100 and best_patch_vv is not None:
        lookalike_result = _lookalike_classifier.classify(best_patch_vv)
        logger.info(
            "Look-alike check: score=%.3f label=%s passed=%s",
            lookalike_result["lookalike_score"],
            lookalike_result["lookalike_label"],
            lookalike_result["lookalike_passed"],
        )
    else:
        lookalike_result = {
            "lookalike_score": None,
            "lookalike_label": None,
            "lookalike_passed": None,
        }
    # ────────────────────────────────────────────────────────────────────────

    # ── Phase D: Wind context (ADDED HERE SAFELY) ───────────────────────────
    wind = _wind_context.get_context(
        lat=0.0 if geojson_polygon is None else geojson_polygon["coordinates"][0][0][1],
        lon=0.0 if geojson_polygon is None else geojson_polygon["coordinates"][0][0][0],
        timestamp=None  # replace later with Sentinel metadata if available
    )

    drift_geojson = None
    if wind["drift_vector"]:
        drift_geojson = drift_arrow_geojson(
            0.0 if geojson_polygon is None else geojson_polygon["coordinates"][0][0][1],
            0.0 if geojson_polygon is None else geojson_polygon["coordinates"][0][0][0],
            wind["drift_vector"],
            hours=24
        )
    # ────────────────────────────────────────────────────────────────────────

    result = {
        "confidence": round(confidence * 100, 2),
        "area_km2": area_km2,
        "spill_pixels": int(spill_pixels),
        "polygon": geojson_polygon,
        "detected": spill_pixels > 100,
        "scene": Path(vv_path).name,

        # Phase C
        "lookalike_score": lookalike_result["lookalike_score"],
        "lookalike_label": lookalike_result["lookalike_label"],
        "lookalike_passed": lookalike_result["lookalike_passed"],

        # Phase D (Wind integration)
        "wind_speed_ms": wind["wind_speed_ms"],
        "wind_direction_deg": wind["wind_direction_deg"],
        "wind_u": wind["wind_u"],
        "wind_v": wind["wind_v"],
        "sar_validity": wind["sar_validity"],
        "sar_validity_detail": wind["sar_validity_detail"],
        "lookalike_wind_risk": wind["lookalike_wind_risk"],
        "lookalike_wind_note": wind["lookalike_wind_note"],
        "drift_bearing_deg": wind["drift_vector"]["bearing_deg"] if wind["drift_vector"] else None,
        "drift_speed_ms": wind["drift_vector"]["speed_ms"] if wind["drift_vector"] else None,
        "drift_24h_km": wind["drift_vector"]["24h_km"] if wind["drift_vector"] else None,
        "drift_geojson": json.dumps(drift_geojson) if drift_geojson else None,
        "wind_fetched_at": wind["wind_fetched_at"],
        "wind_data_source": wind["wind_data_source"],
    }

    return result


if __name__ == "__main__":
    from backend.preprocess import find_bands
    import zipfile

    scenes_dir = Path("data/scenes")

    for zf in scenes_dir.glob("*.zip"):
        extract_dir = scenes_dir / zf.stem
        if not extract_dir.exists():
            print(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(scenes_dir)

    safe_dirs = list(scenes_dir.glob("*.SAFE"))
    if not safe_dirs:
        print("No .SAFE scene found in data/scenes/")
    else:
        scene = safe_dirs[0]
        print(f"Running detection on: {scene.name}")
        vv_path, vh_path = find_bands(str(scene))

        if vv_path:
            result = run_detection(vv_path, vh_path)
            print("\n--- DETECTION RESULT ---")
            print(result)
        else:
            print("Could not find VV band in scene.")