# Soil Erosion Modelling – LOAO

This repository provides a fully reproducible workflow for modelling soil erosion using four machine-learning approaches:

- **Patch-based Convolutional Neural Network (CNN)**
- **Deep Neural Network (DNN)**
- **Single-Hidden-Layer Network (SNN)**
- **Random Forest (RF)**

All models are trained and evaluated using **Leave-One-Area-Out (LOAO) validation**, ensuring that performance is assessed on spatially unseen terrain.  
After training, **area-specific trained models** are used to generate full-resolution GeoTIFF prediction maps.

The grid-search pipeline can be found in `grid_search/` 

---

## Repository Structure

```
project_root/
├── config.yaml                 
├── data/                       
├── output/
│   ├── models/                 
│   └── predictions/            
├── modules/
│   ├── io_utils.py             
│   ├── cnn_utils.py            
│   ├── patches.py              
│   ├── models.py               
│   └── data_prep.py            
├── train.py                    
├── predict.py                  
├── grid_search/                
├── requirements.txt
```

---

## Workflow Overview

### **1. LOAO Training (`train.py`)**

For each study area:

- Train on **all other areas**
- Validate on the **held-out area**
- Record metrics (RMSE, MAE, F1, spatial correlation)
- Save trained model + scaler:

```
output/models/<model_type>/holdout_<AREA>/
    ├── model.h5 or model.pkl
    └── scaler.pkl
```

### **2. Inference (`predict.py`)**

For each area:

- Loads the corresponding LOAO-trained model  
- Applies scaling & patch extraction  
- Generates a full GeoTIFF map:

```
output/predictions/<model_type>/<AREA>.tif
```

---

## Configuration (`config.yaml`)

```yaml
data_dir: data/
output_dir: output/

study_areas:
  folder_mode: false
  list:
    - Adenstedt
    - Barum
    - Brueggen
    - Kleinilde
    - Kueingdorf
    - Lamspringe
    - Nette

model_type: cnn

models:
  cnn:
    lr: 1e-5
    patch_size: 7
    conv_layers: 4
    filters: 256
    l2_conv: 1e-4
    l2_dense: 1e-4
    dropout: 0.5
    kernel_size: 3

  dnn:
    lr: 1e-4
    layers: 3
    units: 256
    dropout: 0.25
    l2_dense: 1e-4

  snn:
    lr: 1e-3
    units: 128
    l2_dense: 1e-2

  rf:
    n_estimators: 2000
    max_depth: 20
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: log2

bands:
  - ...
```

---
