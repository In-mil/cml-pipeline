cat > src/train.py << 'PY'
import argparse, os, json, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import mlflow
from yellowbrick.regressor import ResidualsPlot
import matplotlib.pyplot as plt

def load_data():
    # lies DVC-Datei; echte Daten liegen nach `dvc pull` lokal
    df = pd.read_parquet("data/green_tripdata_2025-01.parquet")
    # einfache, robuste Feature-Auswahl
    cols = [c for c in df.columns if df[c].dtype != "object"]
    df = df[cols].dropna()
    y = df[cols[0]]
    X = df[cols[1:]]
    return X, y

def train(cml_run: bool):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("cml-green-taxi")

    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2
        )
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        rmse = mean_squared_error(yte, preds, squared=False)
        r2 = r2_score(yte, preds)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        if cml_run:
            pathlib.Path("metrics.txt").write_text(f"RMSE: {rmse:.4f}\nR2: {r2:.4f}\n")
            viz = ResidualsPlot(model)
            viz.fit(Xtr, ytr)
            viz.score(Xte, yte)
            viz.poof(outpath="residuals.png")  # speichert Plot
            # Fallback: falls poof nicht verfügbar
            try:
                plt.close('all')
            except Exception:
                pass

            # kleiner JSON-Schnappschuss
            pathlib.Path("metrics.json").write_text(json.dumps({"rmse": rmse, "r2": r2}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cml-run", dest="cml_run", action="store_true")
    p.add_argument("--no-cml-run", dest="cml_run", action="store_false")
    p.set_defaults(cml_run=False)
    args = p.parse_args()
    train(args.cml_run)
PY
git add src/train.py
git commit -m "Add train.py with --cml-run support"
