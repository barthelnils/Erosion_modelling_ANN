# Soil Erosion Modelling

This repository provides a reproducible workflow for predicting soil erosion rates using multiple machine learning models (convolutional (CNN), deep (DNN), single-hidden-layer (SNN) neural network, and Random Forest(RF)). It includes data preprocessing, model training with cross‑validation, evaluation metrics, and inference to generate GeoTIFF prediction maps.

---

## Repository Structure
```
project_root/
├── config.yaml         # Data paths, study areas, model settings
├── data/               # Input GeoTIFFs (small samples or pointers)
├── output/             # Generated prediction maps & metrics
├── modules/            # Core code modules
│   ├── data_prep.py    # I/O & preprocessing functions
│   ├── models.py       # Model definitions (CNN, DNN, SNN, RF)
│   ├── evaluation.py   # Metrics & permutation importance
│   └── inference.py    # Inference & GeoTIFF export
├── train.py            # Cross‑validation training script
├── predict.py          # Inference script to generate .tif maps
├── requirements.txt    # Python dependencies
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo_url>
cd project_root
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure parameters
Edit `config.yaml` to point to your `data_dir`, set `output_dir`, list study areas or enable `folder_mode`, choose bands, and select model hyperparameters.

### 4. Prepare your data
Place your input GeoTIFFs in the `data/` folder. Each file should contain raster bands named in `config.yaml` and a target band (first band) with soil erosion values.


## Usage

### A. Training
Run cross‑validation training and save artifacts:
```bash
python train.py --config config.yaml
```
- Outputs:
  - `output/<model>_model.h5` or `.pkl` (trained model)
  - `output/scaler.pkl` (feature scaler)
  - `output/metrics_importances.pkl` (CV metrics & importances)

### B. Inference
Generate prediction maps for all study areas:
```bash
python predict.py --config config.yaml
```
- Outputs:
  - GeoTIFFs in `output/` named `<model>_predictions_<area>.tif`


## Configuration (`config.yaml`)
Key parameters:
- `data_dir`: folder with input `.tif` files
- `output_dir`: folder for model & prediction outputs
- `study_areas.folder_mode`: `true` to auto‑detect all `.tif` in `data_dir`
- `study_areas.list`: explicit list of area names
- `bands`: list of raster band names to load
- `model.type`: `cnn`, `dnn`, `snn`, or `rf`
- `model.epochs`, `model.batch_size`, `model.threshold`
- `cv_folds`: number of CV folds



