import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter
from pathlib import Path
import os

PATCH_SIZE = 256
PATCHES_DIR = "data/patches"
SCENES_DIR = "data/scenes"
CHUNK_SIZE = 2048  # process 2048x2048 pixels at a time

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

def preprocess_scene(vv_path: str, vh_path: str = None, scene_id: str = None):
    """
    Process scene in chunks to avoid memory errors.
    Reads CHUNK_SIZE x CHUNK_SIZE blocks, tiles into 256x256 patches.
    """
    out_dir = None
    if scene_id:
        out_dir = Path(PATCHES_DIR) / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

    patch_count = 0

    with rasterio.open(vv_path) as vv_src:
        height = vv_src.height
        width = vv_src.width
        meta = vv_src.meta
        transform = vv_src.transform
        print(f"Scene size: {height} x {width} pixels")
        print(f"Processing in {CHUNK_SIZE}x{CHUNK_SIZE} chunks...")

        vh_src = None
        if vh_path and os.path.exists(vh_path):
            vh_src = rasterio.open(vh_path)

        all_patches = []

        for row_off in range(0, height - CHUNK_SIZE + 1, CHUNK_SIZE):
            for col_off in range(0, width - CHUNK_SIZE + 1, CHUNK_SIZE):
                window = Window(col_off, row_off, CHUNK_SIZE, CHUNK_SIZE)

                # Read VV chunk
                vv_chunk = vv_src.read(1, window=window).astype(np.float32)
                vv_db = to_db(vv_chunk)
                vv_filtered = lee_filter(vv_db)
                vv_norm = normalize(vv_filtered)

                # Read VH chunk
                if vh_src:
                    vh_chunk = vh_src.read(1, window=window).astype(np.float32)
                    vh_db = to_db(vh_chunk)
                    vh_filtered = lee_filter(vh_db)
                    vh_norm = normalize(vh_filtered)
                else:
                    vh_norm = vv_norm.copy()

                # Tile this chunk into 256x256 patches
                h, w = vv_norm.shape
                for i in range(0, h - PATCH_SIZE + 1, PATCH_SIZE):
                    for j in range(0, w - PATCH_SIZE + 1, PATCH_SIZE):
                        vv_tile = vv_norm[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        vh_tile = vh_norm[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        patch = np.stack([vv_tile, vh_tile], axis=0)  # (2, 256, 256)

                        patch_info = {
                            "patch": patch,
                            "row": row_off + i,
                            "col": col_off + j,
                            "idx": patch_count
                        }

                        if out_dir:
                            out_path = out_dir / f"patch_{patch_count:04d}_r{row_off+i}_c{col_off+j}.npy"
                            np.save(out_path, patch)

                        all_patches.append(patch_info)
                        patch_count += 1

                print(f"  Chunk ({row_off},{col_off}) done — {patch_count} patches so far")

                # Stop early after enough patches for training
                if patch_count >= 500:
                    print("Reached 500 patches — stopping early (enough for training).")
                    if vh_src:
                        vh_src.close()
                    return all_patches, meta, transform

        if vh_src:
            vh_src.close()

    print(f"\nPreprocessing complete. Total patches: {patch_count}")
    return all_patches, meta, transform

def find_bands(scene_dir: str):
    scene_path = Path(scene_dir)
    vv, vh = None, None
    for f in scene_path.rglob("*"):
        name = f.name.lower()
        if ("vv" in name) and (name.endswith(".tiff") or name.endswith(".tif") or name.endswith(".img")):
            vv = str(f)
        if ("vh" in name) and (name.endswith(".tiff") or name.endswith(".tif") or name.endswith(".img")):
            vh = str(f)
    return vv, vh

if __name__ == "__main__":
    import zipfile

    scenes_dir = Path(SCENES_DIR)
    zip_files = list(scenes_dir.glob("*.zip"))

    for zf in zip_files:
        extract_dir = scenes_dir / zf.stem
        if not extract_dir.exists():
            print(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(scenes_dir)
            print("Extracted.")

    all_safe = list(scenes_dir.glob("*.SAFE"))
    if all_safe:
        scene = all_safe[0]
        print(f"\nPreprocessing: {scene.name}")
        vv_path, vh_path = find_bands(str(scene))
        if vv_path:
            patches, meta, transform = preprocess_scene(
                vv_path, vh_path, scene_id=scene.stem
            )
            print(f"\nDone. Total patches: {len(patches)}")
            print(f"Patch shape: {patches[0]['patch'].shape}")
        else:
            print("Could not find VV band.")
    else:
        print("No .SAFE directory found.")