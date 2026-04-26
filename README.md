<div align="center">

```
███████╗██████╗ ███████╗ ██████╗████████╗██████╗  █████╗
██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
███████╗██████╔╝█████╗  ██║        ██║   ██████╔╝███████║
╚════██║██╔═══╝ ██╔══╝  ██║        ██║   ██╔══██╗██╔══██║
███████║██║     ███████╗╚██████╗   ██║   ██║  ██║██║  ██║
╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
```

### AI-Powered Oil Spill Detection Platform — Niger Delta Surveillance

![Status](https://img.shields.io/badge/Status-Active%20Development-00d4aa?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Sentinel](https://img.shields.io/badge/Sentinel--1-SAR%20Radar-1a73e8?style=for-the-badge)
![ERA5](https://img.shields.io/badge/ERA5-Wind%20Context-f5a623?style=for-the-badge)
![Sentinel2](https://img.shields.io/badge/Sentinel--2-Optical%20Cross--Validation-4a9eff?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)

</div>

---

## 🛰️ What is Spectra?

Spectra is an end-to-end satellite-based oil spill detection platform built for industrial use. It ingests Sentinel-1 SAR radar scenes over the Niger Delta, runs a multi-stage AI detection pipeline, cross-validates findings against Sentinel-2 optical data and ERA5 wind data, and dispatches real-time email alerts to operators, environmental agencies, and legal teams.

> **The problem:** Oil spills in the Niger Delta are frequently undiscovered for days or weeks. Existing tools are expensive, require expert operators, and produce too many false positives to be legally defensible. Spectra addresses all three.

Spectra is not a demo. It is built to the evidence standard required by environmental regulators and legal teams — every detection is stored with a full audit trail combining SAR backscatter analysis, look-alike classification, wind context, and optical confirmation.

---

## 🧠 How It Works

The detection pipeline runs in sequence, with each stage adding a layer of confidence:

| Stage | Module | What It Does |
|---|---|---|
| 1 | `preprocess.py` | Ingests Sentinel-1 GRD scene, applies speckle filter, normalises to 256×256 patches |
| 2 | `detect.py` → U-Net | ResNet34-encoder U-Net segments spill pixels from SAR backscatter (VV + VH) |
| 3 | `lookalike_classifier.py` | Lightweight MobileNet CNN classifies: is this oil or a look-alike (algae, calm water, wind shadow)? |
| 4 | `wind_context.py` | Fetches ERA5 10m wind at detection location and time — validates SAR detection window, computes 24h drift vector |
| 5 | `optical_validator.py` | Pulls Sentinel-2 L2A scene within ±3 days, computes OSI / SWIRI / NDWI spectral indices, generates RGB and false-colour thumbnails |
| 6 | `alerts.py` | If all gates pass, fires Resend email alert to configured recipients with full detection evidence |
| 7 | `database.py` | Permanently stores every detection with all fields for legal audit trail |

---

## ⚙️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.11 + FastAPI** | Backend REST API and scan pipeline orchestration |
| **PyTorch + segmentation-models-pytorch** | U-Net with ResNet34 encoder for SAR pixel segmentation |
| **SQLite + SQLAlchemy** | Persistent detection database with full audit trail |
| **Copernicus CDSE** | Sentinel-1 SAR scene search and download |
| **ERA5 / CDS API** | Hourly 10m wind data (u10, v10) for wind context layer |
| **Sentinel-2 L2A** | Optical cross-validation: OSI, SWIRI, NDWI spectral indices + RGB thumbnails |
| **Rasterio + NumPy + Pillow** | Band extraction, spectral index computation, thumbnail generation |
| **Resend** | Transactional email alert dispatch |
| **Leaflet.js + Leaflet.draw** | Interactive geospatial map, Watch Zone polygon drawing, drift arrow rendering |
| **HTML / CSS / Vanilla JS** | Frontend dashboard — no framework dependencies |

---

## ✨ Key Features

- 🎯 **U-Net SAR Segmentation** — ResNet34-encoder U-Net trained on the Kaggle oil spill SAR dataset. Best IoU: 0.7066. Produces pixel-level spill mask and GeoJSON polygon per detection.
- 🔍 **Look-alike Classifier** — MobileNet-style CNN runs after U-Net and asks: *is this actually oil?* Kills false positives from algae, calm water, wind shadows, and rain cells. Outputs `lookalike_score` (0–1).
- 🌬️ **ERA5 Wind Context** — Fetches real wind speed and direction at detection time and location. Assesses SAR detection validity window (2–10 m/s), flags look-alike risk, and computes a 3% empirical drift vector showing where the slick will move over 6h / 12h / 24h horizons. Rendered as an amber arrow on the Leaflet map.
- 🛰️ **Sentinel-2 Optical Cross-Validation** — Pulls the nearest-in-time S2 L2A scene and computes three peer-reviewed spectral indices. Produces a true-colour and SWIR false-colour thumbnail for each detection. Verdict: `confirmed` / `unconfirmed` / `inconclusive`.
- 📍 **Watch Zones** — Operators draw permanent polygons over oil fields and pipeline corridors on the map. Zones are stored to the database and every scan checks detections against zone boundaries.
- 🔔 **Alert Dispatch** — Animated dispatch flow modal lets operators add email recipients and fire alerts. Resend integration confirmed working. Alert log panel tracks every dispatch.
- 🗄️ **Legal Audit Trail** — Every detection stored permanently in SQLite with confidence score, area, GeoJSON polygon, lookalike score, wind data, spectral indices, optical thumbnails, and timestamps.
- 🗺️ **Multi-Layer Geospatial Map** — Spill polygons, pipeline corridors, historical incidents, vegetation stress zones, wind drift arrows, SAR backscatter overlay — all independently toggleable.

---

## 🚀 Running Locally

### Prerequisites

```bash
# Python 3.11+ required
pip install fastapi uvicorn sqlalchemy torch segmentation-models-pytorch \
            rasterio pillow shapely cdsapi xarray netcdf4 scipy \
            --break-system-packages
```

### Environment Variables

```bash
export CDSE_USER=your@email.com
export CDSE_PASSWORD=yourpassword
export RESEND_API_KEY=re_your_key_here
```

### ERA5 Wind Setup (one-time)

```bash
# 1. Register at https://cds.climate.copernicus.eu
# 2. Accept ERA5 dataset licence at:
#    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
# 3. Create ~/.cdsapirc:

nano ~/.cdsapirc
```

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR-PERSONAL-ACCESS-TOKEN>
```

### Start the Platform

```bash
# Terminal 1 — Backend
cd /workspaces/spectra/spectra
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd /workspaces/spectra/spectra/frontend
python -m http.server 3000
```

### Copernicus WAF Workaround (GitHub Codespaces)

> Codespace IPs are blocked by the Copernicus WAF for direct scene downloads. Scene **search** works. Download scenes manually via [browser.dataspace.copernicus.eu](https://browser.dataspace.copernicus.eu) and place them in `data/scenes/`. Spectra auto-detects scenes by timestamp matching.

---

## 📁 File Structure

```
spectra/
├── backend/
│   ├── main.py                  # FastAPI app — all endpoints, scan pipeline
│   ├── database.py              # SQLAlchemy models: Detection, WatchZone, AlertLog
│   ├── detect.py                # Detection loop — U-Net → Lookalike → Wind → Optical
│   ├── train.py                 # U-Net training script
│   ├── preprocess.py            # SAR preprocessing, 256×256 patch extraction
│   ├── lookalike_classifier.py  # Phase C — MobileNet look-alike CNN
│   ├── wind_context.py          # Phase D — ERA5 wind fetch + drift vector
│   ├── optical_validator.py     # Phase E — Sentinel-2 spectral cross-validation
│   ├── alerts.py                # Resend email alert dispatch
│   ├── satellite_query.py       # Copernicus CDSE scene search
│   └── satellite_download.py    # Scene download with WAF handling
├── frontend/
│   └── index.html               # Complete single-file dashboard (Phases A–E)
├── models/
│   ├── spectra_model.pth        # U-Net weights (94MB, best IoU 0.7066)
│   └── lookalike_model.pth      # Look-alike classifier weights
├── data/
│   ├── spectra.db               # SQLite database
│   ├── scenes/                  # Sentinel-1 and Sentinel-2 .SAFE scenes
│   └── lookalike_dataset/       # Training data for look-alike classifier
│       ├── oil/
│       └── non_oil/
└── README.md
```

---

## 📊 Model Performance

| Model | Architecture | Dataset | Best Metric |
|---|---|---|---|
| U-Net (SAR segmentation) | ResNet34 encoder, 2ch input | Kaggle Oil Spill SAR (~1000 images) | IoU: **0.7066** |
| Look-alike Classifier | MobileNet-style CNN, ~480K params | Kaggle dataset (oil / non-oil split) | Val accuracy: **~87–90%** |
| Optical Validator | Spectral indices (no model) | ERA5 + Sentinel-2 L2A | SWIRI specificity: **>95%** |

---

## 🗺️ Roadmap

### Completed

- [x] Phase A — Watch Zones (operator draws polygons, stored to DB permanently)
- [x] Phase B — Database Persistence (full SQLite audit trail)
- [x] Phase C — Look-alike Classifier (MobileNet CNN, trained and integrated)
- [x] Phase D — Wind Context Layer (ERA5 API, drift vector, SAR validity gate)
- [x] Phase E — Sentinel-2 Optical Cross-Validation (OSI, SWIRI, NDWI, RGB thumbnails)

### In Progress / Planned

- [ ] Phase F — Report Generation (professional PDF output for regulatory submission)
- [ ] Phase G — AIS Vessel Attribution (cross-reference ship tracking data to identify source vessels)
- [ ] Phase H — Retrain with UNet++ on expanded dataset for higher IoU

---

## 🎯 Built For

Spectra is built for industrial deployment by:

- **Environmental NGOs** — automated surveillance of high-risk pipeline corridors
- **Government regulators** — legally defensible detection evidence with full audit trail
- **Legal teams** — SAR polygon + wind context + optical thumbnail = three-sensor evidence package
- **Oil companies** — early spill detection before public or regulatory discovery

---

## 👤 Author

**David Iwuoha**
[github.com/david-iwuoha/spectra](https://github.com/david-iwuoha/spectra)

> *Spectra is under active development. The detection pipeline (Phases A–E) is production-ready. Phases F–H are in the roadmap.*

---

<div align="center">

![Sentinel-1](https://img.shields.io/badge/Powered%20by-Sentinel--1%20SAR-1a73e8?style=flat-square)
![Sentinel-2](https://img.shields.io/badge/Cross--Validated%20by-Sentinel--2%20Optical-4a9eff?style=flat-square)
![ERA5](https://img.shields.io/badge/Wind%20Context-ERA5%20ECMWF-f5a623?style=flat-square)
![Built in](https://img.shields.io/badge/Built%20in-GitHub%20Codespaces-24292e?style=flat-square&logo=github)

</div>