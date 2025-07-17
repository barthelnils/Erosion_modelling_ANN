import os
import yaml

from modules.inference import run_inference

def main(config_path='config.yaml'):
    # Load configuration
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg['data_dir']
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Determine study areas
    if cfg['study_areas']['folder_mode']:
        areas = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith('.tif')]
    else:
        areas = cfg['study_areas']['list']

    bands = cfg['bands']
    model_type = cfg['model']['type']

    # Derive model and scaler paths
    model_ext = 'h5' if model_type in ('cnn', 'dnn', 'snn') else 'pkl'
    model_path = os.path.join(output_dir, f"{model_type}_model.{model_ext}")
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    # Run inference and save GeoTIFFs
    run_inference(
        model_path=model_path,
        scaler_path=scaler_path,
        areas=areas,
        bands=bands,
        output_dir=output_dir,
        model_type=model_type
    )

if __name__ == '__main__':
    main()
