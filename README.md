# Titanic Survival Prediction — Logistic Regression 

This project implements a complete, reproducible data-science workflow for the Titanic survival problem using Logistic Regression in a scikit-learn pipeline. The notebook is organized by the DS lifecycle: setup, data loading, cleaning, EDA, feature engineering, modeling, evaluation, diagnostics, saving artifacts, and inference.

## Contents
- `Logistic Regression.ipynb` — End-to-end workflow notebook
- `requirements.txt` — Python dependencies
- `artifacts/` — Saved model and metrics after running the notebook
  - `model.joblib`
  - `feature_names.json`
  - `test_metrics.json`

## Environment & Setup
- Python: Tested with Python 3.13 (works with modern 3.x)
- Create and activate a virtual environment (optional if using VS Code’s notebook kernel):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer running inside VS Code’s interactive notebooks, select the `.venv` kernel and install when prompted.

## Data
The notebook attempts to load the Titanic dataset from seaborn (`sns.load_dataset("titanic")`). If that fails, it will look for a local CSV in one of these paths (first found wins):
- `train.csv`
- `data/train.csv`
- `dataset/train.csv`

Place a compatible Titanic CSV in any of the above paths if seaborn’s dataset is unavailable (e.g., air-gapped environments).

## How to Run
1. Open `Logistic Regression.ipynb` in VS Code (or Jupyter).
2. Run cells top-to-bottom in order. The sections follow this workflow:
   1) Setup & Configuration
   2) Data Loading
   3) Data Cleaning
   4) Exploratory Data Analysis (EDA)
   5) Feature Engineering
   6) Train/Validation/Test Split
   7) Preprocessing + Model Pipeline (Logistic Regression)
   8) Baseline Training and Cross-Validation
   9) Hyperparameter Tuning (GridSearchCV)
   10) Final Test Evaluation
   11) Diagnostics (confusion matrix, coefficients)
   12) Save Artifacts
   13) Inference Example

When complete, the notebook writes artifacts to `./artifacts/`.

## Outputs
- `artifacts/model.joblib` — Trained scikit-learn `Pipeline` (preprocessing + LogisticRegression)
- `artifacts/feature_names.json` — Feature names after preprocessing
- `artifacts/test_metrics.json` — Final test metrics (accuracy, precision, recall, F1, ROC-AUC)

## Reproducibility
- `random_state = 42` is used for splitting and CV.
- The pipeline uses `OneHotEncoder` + `StandardScaler` with imputers; hyperparameters are tuned via `GridSearchCV`.

## Troubleshooting
- Import errors: ensure the environment is active and run `pip install -r requirements.txt`.
- Dataset not found: place a Titanic CSV at `train.csv`, `data/train.csv`, or `dataset/train.csv`.
- Kernel state issues: restart the kernel and re-run all cells after installing packages.

## Extending
- Try alternative models (RandomForest, Gradient Boosting)
- Add probability calibration (`CalibratedClassifierCV`) if needed
- Track experiments with MLflow or Weights & Biases
- Package inference with FastAPI or Flask for deployment
