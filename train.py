import os
import yaml
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import modules.data_prep as dp
from modules.data_prep import prepare_dataset, normalize_features
from modules.models import create_cnn, create_dnn, create_snn, create_rf
from tensorflow.keras.callbacks import EarlyStopping

def calculate_metrics(y_true, y_pred, threshold=0.25):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)

    bin_t = (y_true >= threshold).astype(int)
    bin_p = (y_pred >= threshold).astype(int)

    # For multi-class, re-use your categorize function if needed,
    # but here we’ll skip finer bins and just keep binary metrics:
    accuracy  = accuracy_score(bin_t, bin_p)
    precision = precision_score(bin_t, bin_p, zero_division=0)
    recall    = recall_score(bin_t, bin_p, zero_division=0)
    f1        = f1_score(bin_t, bin_p, zero_division=0)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main(config_path='config.yaml'):
    # 1) Load config
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)

    data_dir   = cfg['data_dir']
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 2) Override DATA_DIR for data_prep
    dp.DATA_DIR = data_dir

    # 3) Areas & bands
    if cfg['study_areas']['folder_mode']:
        areas = [os.path.splitext(f)[0]
                 for f in os.listdir(data_dir) if f.endswith('.tif')]
    else:
        areas = cfg['study_areas']['list']
    bands    = cfg['bands']
    mcfg     = cfg['model']
    folds    = cfg.get('cv_folds', 5)

    # 4) Load + mask + flatten
    info, X, y = prepare_dataset(areas, bands)
    print(f"Loaded {X.shape[0]} samples; y-range {y.min():.3f}-{y.max():.3f}")

    # 5) Scale
    Xs, scaler = normalize_features(X)

    # 6) Reshape for NN
    is_nn = mcfg['type'] in ('cnn','dnn','snn')
    if is_nn:
        if mcfg['type']=='cnn':
            Xs_in = Xs.reshape(-1,1,1,len(bands))
        else:
            Xs_in = Xs
    else:
        Xs_in = Xs

    # 7) Cross‑val
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    all_metrics = []

    for i,(tr,te) in enumerate(kf.split(Xs_in),1):
        print(f"\nFold {i}/{folds}")
        Xtr, Xte = Xs_in[tr], Xs_in[te]
        ytr, yte = y[tr], y[te]

        # instantiate
        mtype = mcfg['type']
        params= mcfg.get('params',{})
        if mtype=='cnn':
            model = create_cnn(Xtr.shape[1:], **params)
        elif mtype=='dnn':
            model = create_dnn(Xtr.shape[1], **params)
        elif mtype=='snn':
            model = create_snn(Xtr.shape[1], **params)
        else:
            model = create_rf(**params)

        # train
        if is_nn:
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(
                Xtr, ytr,
                validation_data=(Xte,yte),
                epochs=mcfg.get('epochs',50),
                batch_size=mcfg.get('batch_size',256),
                callbacks=[es],
                verbose=1
            )
            preds = model.predict(Xte).flatten()
        else:
            model.fit(Xtr,ytr)
            preds = model.predict(Xte)

        # eval
        mets = calculate_metrics(yte,preds, threshold=mcfg.get('threshold',0.25))
        print("  Metrics:", mets)
        all_metrics.append(mets)

    # 8) Aggregate metrics
    mean_metrics = {
        k: np.mean([m[k] for m in all_metrics])
        for k in all_metrics[0]
    }
    print("\nMean CV Metrics:")
    for k,v in mean_metrics.items():
        print(f"  {k:>8}: {v:.4f}")

    # 9) Save final model + scaler + metrics
    ext = 'h5' if is_nn else 'pkl'
    model_path  = os.path.join(output_dir, f"{mtype}_model.{ext}")
    if is_nn:
        model.save(model_path)
    else:
        with open(model_path,'wb') as f: pickle.dump(model,f)

    with open(os.path.join(output_dir,'scaler.pkl'),'wb') as f:
        pickle.dump(scaler,f)

    metrics_path = os.path.join(output_dir,'metrics.pkl')
    with open(metrics_path,'wb') as f:
        pickle.dump(mean_metrics,f)

    print("\nArtifacts saved:")
    print(" - Model: ", model_path)
    print(" - Scaler:", os.path.join(output_dir,'scaler.pkl'))
    print(" - Metrics:", metrics_path)

if __name__=='__main__':
    main()
