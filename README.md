# Surface Roughness Machine Learning Model

GitHub: [nhase13-spec/surface-roughness-machine-learning-model](https://github.com/nhase13-spec/surface-roughness-machine-learning-model)

Predict **CNC surface roughness (Ra)** from **programmed feed and spindle settings** using a small public milling dataset.

**Data:** [Zenodo 21518387](https://zenodo.org/records/21518387) (CC BY 4.0) — this repo includes only the two summary CSVs needed for training (see `data/raw/dataset/DATASET.txt`).

## Quick start

```powershell
cd surface_roughness_machine_learning_model
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

## Project layout

| Folder / file | Purpose |
|---------------|---------|
| `config.yaml` | Target column, paths, CV settings |
| `run_pipeline.py` | Run training and save results |
| `src/` | Load data, features, train models, plots |
| `tests/` | Checks for data leakage and grouped CV |
| `data/raw/dataset/` | Input CSV files |
| `results/` | Created after you run the pipeline (figures, metrics) |
| `models/` | Saved models after you run the pipeline |

## Models compared

Dummy mean, Ridge, Random Forest, XGBoost, Gaussian Process — with **leave-one-segment-out** cross-validation (15 segments, 45 profiles).
