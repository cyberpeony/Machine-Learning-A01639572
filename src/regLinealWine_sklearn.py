"""
Regresión (framework) con scikit-learn sobre Wine Quality
- Pipeline: StandardScaler + (LinearRegression o Ridge)
- Métricas: R2, MSE, RMSE, MAE + baseline
- Usando crossvalidation (k-folds) y grid search de alpha (para ridge, l2)

Training simple (linear):
  python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality \
    --model-path model_sklearn.pkl --modelo linear --test-size 0.2 --seed 42
Training simple (ridge):
  python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality \
    --model-path model_sklearn.pkl --modelo ridge --alpha 0.1 --test-size 0.2 --seed 42
Crossvalidation + grid (ridge):
  python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality \
    --model-path model_sklearn.pkl --modelo ridge --cv 5 --grid_alpha 0.0 0.01 0.1 1.0 --seed 42
Predicción:
  python src/regLinealWine_sklearn.py predict --csv data/winequality-red.csv --model-path model_sklearn.pkl \
    --out-csv predicciones_sklearn.csv
"""

import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyRegressor


def rmse(y, y_pred): return float(np.sqrt(mean_squared_error(y, y_pred)))

def build_pipeline(modelo: str, incluir_intercept: bool, alpha: float | None = None) -> Pipeline:
    if modelo == "linear":
        model = LinearRegression(fit_intercept=incluir_intercept)
    elif modelo == "ridge":
        a = 1.0 if alpha is None else float(alpha)
        model = Ridge(alpha=a, fit_intercept=incluir_intercept, random_state=None)
    else:
        raise ValueError("modelo debe ser 'linear' o 'ridge'")
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

def train_cmd(a):
    datos = pd.read_csv(a.csv, sep=",").dropna().reset_index(drop=True)
    if a.target not in datos.columns:
        print(f"error: el target no está en el csv", file=sys.stderr); sys.exit(1)
    y_target = datos[a.target].astype(float).to_numpy()
    X_pred = datos.drop(columns=[a.target]).astype(float)

    # opt 1: crossval + grid (ridge)
    if a.cv and a.cv >= 2:
        if a.modelo != "ridge":
            print("Nota --> --cv está pensado para 'ridge' (grid de alpha), se continuará con ridge", file=sys.stderr)
            a.modelo = "ridge"

        pipe = build_pipeline("ridge", incluir_intercept=(not a.no_intercept))
        param_grid = {"model__alpha": [0.0] if (not a.grid_alpha) else [float(x) for x in a.grid_alpha]}
        cv = KFold(n_splits=a.cv, shuffle=True, random_state=a.seed)
        grid = GridSearchCV(pipe, param_grid=param_grid, scoring="r2", cv=cv, n_jobs=None, refit=True, return_train_score=False)
        grid.fit(X_pred, y_target)

        # resumen de crossval
        print(f"Resultados {a.cv}-fold (media +- desv. estándar)")
        means = grid.cv_results_["mean_test_score"]
        stds  = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for m, s, p in zip(means, stds, params):
            alpha = p["model__alpha"]
            print(f"alpha={alpha:g}  R2={m:.4f}±{s:.4f}")
        best_alpha = grid.best_params_["model__alpha"]
        best_r2 = grid.best_score_
        print(f"\nMejor alpha por R2 promedio: {best_alpha:g} (R2={best_r2:.4f})")

        # training final c/todo el dataset y alpha más eficiente
        final_pipe = build_pipeline("ridge", incluir_intercept=(not a.no_intercept), alpha=best_alpha)
        final_pipe.fit(X_pred, y_target)
        joblib.dump(final_pipe, a.model_path)
        print(f"\nModelo final (entrenado con todo el dataset) guardado en {a.model_path}")
        return

    # opt 2: train/test simple
    X_train, X_test, y_train, y_test = train_test_split(X_pred, y_target, test_size=a.test_size, random_state=a.seed)
    pipe = build_pipeline(a.modelo, incluir_intercept=(not a.no_intercept), alpha=a.alpha)
    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    # métricas test y train
    print("RESULTADOS TEST")
    print(f"R2:   {r2_score(y_test, y_pred_test):.4f}")
    mse_te = mean_squared_error(y_test, y_pred_test)
    print(f"MSE:  {mse_te:.4f}")
    print(f"RMSE: {rmse(y_test, y_pred_test):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, y_pred_test):.4f}")
    print("\nRESULTADOS TRAIN")
    print(f"R2:   {r2_score(y_train, y_pred_train):.4f}")
    print(f"MSE:  {mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"RMSE: {rmse(y_train, y_pred_train):.4f}")
    print(f"MAE:  {mean_absolute_error(y_train, y_pred_train):.4f}")

    # baseline (media)
    base = DummyRegressor(strategy="mean")
    base.fit(X_train, y_train)
    ybase = base.predict(X_test)
    print("\nBASELINE (media de y_train) TEST")
    print(f"R2:   {r2_score(y_test, ybase):.4f}")
    print(f"MSE:  {mean_squared_error(y_test, ybase):.4f}")
    print(f"RMSE: {rmse(y_test, ybase):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, ybase):.4f}")

    joblib.dump(pipe, a.model_path)
    print(f"\nModelo guardado en {a.model_path}") # modelo entrenado en train

def predict_cmd(a):
    pipe: Pipeline = joblib.load(a.model_path)
    datos = pd.read_csv(a.csv, sep=",").dropna().reset_index(drop=True)
    if "quality" in datos.columns: 
        datos = datos.drop(columns=["quality"])
    X_pred = datos.astype(float)
    y_pred = pipe.predict(X_pred)
    out = pd.DataFrame({"prediction": y_pred})
    print("\nprimeras predicciones:")
    print(out.head(10).to_string(index=False))
    if a.out_csv:
        out.to_csv(a.out_csv, index=False)
        print(f"\npredicciones guardadas en {a.out_csv}")

def main():
    p = argparse.ArgumentParser(description="regresión lineal con framework (scikit-learn) para el dataset Wine Quality")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entrenar el modelo")
    t.add_argument("--csv", required=True); t.add_argument("--target", required=True)
    t.add_argument("--model-path", default="model_sklearn.pkl")
    t.add_argument("--modelo", choices=["linear", "ridge"], default="linear")
    t.add_argument("--alpha", type=float, default=1.0, help="solo para ridge")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--no-intercept", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cv", type=int, default=0, help="k-folds para validación cruzada, >=2 activa crossval")
    t.add_argument("--grid_alpha", nargs="*", type=float, default=None, help="lista de alphas para grid (ridge)")
    t.set_defaults(func=train_cmd)

    p2 = sub.add_parser("predict", help="predecir con un modelo entrenado")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--model-path", default="model_sklearn.pkl")
    p2.add_argument("--out-csv", default=None)
    p2.set_defaults(func=predict_cmd)

    a = p.parse_args(); a.func(a)

if __name__ == "__main__":
    main()
