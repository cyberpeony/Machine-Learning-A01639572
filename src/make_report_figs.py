"""
Splits train/validation/test, cálculo de métricas (R2, MSE, RMSE, MAE), curvas de aprendizaje, y comparación Linear vs Ridge.
Guarda figuras en reports/ y tabla de métricas en reports/metricas.csv
"""

import argparse, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def rmse(y, yhat): return float(np.sqrt(mean_squared_error(y, yhat)))

def evaluate(pipe, X_tr, y_tr, X_val, y_val, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    yhat_tr  = pipe.predict(X_tr)
    yhat_val = pipe.predict(X_val)
    yhat_te  = pipe.predict(X_te)
    out = {
        "train_R2": r2_score(y_tr, yhat_tr),
        "valid_R2": r2_score(y_val, yhat_val),
        "test_R2":  r2_score(y_te, yhat_te),
        "train_MSE": mean_squared_error(y_tr, yhat_tr),
        "valid_MSE": mean_squared_error(y_val, yhat_val),
        "test_MSE":  mean_squared_error(y_te, yhat_te),
        "train_RMSE": rmse(y_tr, yhat_tr),
        "valid_RMSE": rmse(y_val, yhat_val),
        "test_RMSE":  rmse(y_te, yhat_te),
        "train_MAE": mean_absolute_error(y_tr, yhat_tr),
        "valid_MAE": mean_absolute_error(y_val, yhat_val),
        "test_MAE":  mean_absolute_error(y_te, yhat_te),
    }
    return out

def learning_curve(pipe_builder, X_tr, y_tr, X_val, y_val, fracs):
    tr_scores, val_scores = [], []
    n = X_tr.shape[0]
    for f in fracs:
        muestras = max(20, int(n * f))  # al menos 20 muestras
        Xsub = X_tr[:muestras]
        ysub = y_tr[:muestras]
        pipe = pipe_builder()
        pipe.fit(Xsub, ysub)
        tr_scores.append(r2_score(ysub, pipe.predict(Xsub)))
        val_scores.append(r2_score(y_val, pipe.predict(X_val)))
    return np.array(tr_scores), np.array(val_scores)

def main():
    ap = argparse.ArgumentParser(description="genera figuras y métricas para el reporte")
    ap.add_argument("--csv", required=True, help="ruta al csv (winequality-red.csv)")
    ap.add_argument("--target", default="quality")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    datos = pd.read_csv(args.csv, sep=",").dropna().reset_index(drop=True)
    if args.target not in datos.columns:
        raise SystemExit(f"target '{args.target}' no está en el csv")
    y = datos[args.target].astype(float).to_numpy()
    X = datos.drop(columns=[args.target]).astype(float)

    # split 60/20/20: primero test=20%, luego valid=25% de (train_temp) --> 0.6/0.2/0.2
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, shuffle=True
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_temp, y_train_temp, test_size=0.25, random_state=args.seed, shuffle=True
    )

    # pipelines
    def pipe_linear():
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(fit_intercept=True))])
    def pipe_ridge():
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=args.ridge_alpha, fit_intercept=True, random_state=None))])

    # evaluación de modelos
    metricas = []
    for name, builder in [("Linear", pipe_linear), ("Ridge", pipe_ridge)]:
        res = evaluate(builder(), X_train, y_train, X_valid, y_valid, X_test, y_test)
        row = {"model": name, **res}
        metricas.append(row)

    metricas_datos = pd.DataFrame(metricas)
    metricas_datos.to_csv(outdir / "metricas.csv", index=False)
    print("tabla de métricas guardada en metricas.csv")
    print(metricas_datos[["model","train_R2","valid_R2","test_R2"]])

    # curvas de aprendizaje (orden actual de X_train, y_train después del split aleatorio)
    fracs = np.linspace(0.1, 1.0, 10)
    for name, builder, fname in [
        ("linear", pipe_linear, "learning_curve_linear.png"),
        (f"ridge (alpha={args.ridge_alpha:g})", pipe_ridge, "learning_curve_ridge.png"),
    ]:
        tr_scores, val_scores = learning_curve(builder, X_train, y_train, X_valid, y_valid, fracs)
        plt.figure()
        plt.plot(fracs, tr_scores, marker="o", label="R2 train")
        plt.plot(fracs, val_scores, marker="s", label="R2 valid")
        plt.xlabel("fracción de datos de entrenamiento")
        plt.ylabel("R2")
        plt.title(f"curva de aprendizaje - {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()
        print("figura guardada:", outdir / fname)

    # comparación en test, barplot R2 test
    plt.figure()
    labels = metricas_datos["model"].tolist()
    vals = metricas_datos["test_R2"].tolist()
    plt.bar(labels, vals)
    plt.ylabel("R2 (test)")
    plt.title("comparación en test: linear vs ridge")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(outdir / "compare_test_r2.png", dpi=150)
    plt.close()
    print("figura guardada:", outdir / "compare_test_r2.png")

if __name__ == "__main__":
    main()
