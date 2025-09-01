"""
Regresión Lineal desde cero aplicada al dataset Wine Quality
- Optimización con gradiente descendente (regularización l2 penalty)
- Preprocesamiento con estandarización (fit solo con train)

Training:
    python src/regLinealWine.py train --csv data/winequality-red.csv --target quality --model-path model.json --learning_rate 0.05 --num_iters 6000 --l2_penalty 0.0
Predicción:
    python src/regLinealWine.py predict --csv data/winequality-red.csv --model-path model.json --out-csv predicciones.csv
"""

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

# utils
def split_indices(n, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(round(n * test_size))
    return idx[n_test:], idx[:n_test]

def r2(y, preds):
    y = np.asarray(y).ravel(); preds = np.asarray(preds).ravel()
    ss_res = np.sum((y - preds) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def rmse(y, preds):
    y = np.asarray(y).ravel(); preds = np.asarray(preds).ravel()
    return float(np.sqrt(np.mean((y - preds) ** 2)))

def mae(y, preds):
    y = np.asarray(y).ravel(); preds = np.asarray(preds).ravel()
    return float(np.mean(np.abs(y - preds)))

# modelo
class RegLinealWine:
    def __init__(self, learning_rate=0.05, num_iters=5000, l2_penalty=0.0, fit_intercept=True, seed=42, verbose=False):
        self.learning_rate = learning_rate; self.num_iters = num_iters; self.l2_penalty = l2_penalty; self.fit_intercept = fit_intercept
        self.seed = seed; self.verbose = verbose; self.w = None

    def _add_b(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X]) if self.fit_intercept else X

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).reshape(-1, 1); XBias = self._add_b(X)
        rng = np.random.default_rng(self.seed); self.w = rng.normal(0, 0.01, (XBias.shape[1], 1))
        n = XBias.shape[0]

        for it in range(self.num_iters):
            preds = XBias @ self.w
            err = preds - y
            grad = (XBias.T @ err) / n
            if self.l2_penalty > 0:
                reg = self.l2_penalty * self.w
                if self.fit_intercept: reg[0, 0] = 0.0
                grad += reg
            self.w -= self.learning_rate * grad

            if self.verbose and (it % max(1, self.num_iters // 10) == 0 or it == self.num_iters - 1):
                mse = float(np.mean(err ** 2))
                if self.l2_penalty > 0:
                    mse += 0.5 * self.l2_penalty * float(np.sum(self.w[1:] ** 2)) if self.fit_intercept else 0.5 * self.l2_penalty * float(np.sum(self.w ** 2))
                print(f"[{it:5}] MSE={mse:.6f}", file=sys.stderr)

    def predict(self, X):
        X = np.asarray(X, float); XBias = self._add_b(X); return (XBias @ self.w).ravel()

# persistencia
def save_json(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))

# comandos
def train_cmd(args):
    df = pd.read_csv(args.csv, sep=',').dropna().reset_index(drop=True)
    if args.target not in df.columns:
        print(f"error: el target no está en el csv", file=sys.stderr); sys.exit(1)

    y = df[args.target].astype(float).to_numpy()
    X = df.drop(columns=[args.target]).astype(float)

    index_train, index_test = split_indices(len(df), test_size=args.test_size, seed=args.seed)
    X_train, X_test = X.iloc[index_train].copy(), X.iloc[index_test].copy(); y_train, y_test = y[index_train], y[index_test]

    features = X_train.columns.tolist()
    feature_means = X_train.mean(0); feature_stds = X_train.std(0, ddof=0).replace(0.0, 1.0)

    X_train = (X_train - feature_means) / feature_stds
    X_test = X_test.reindex(columns=features, fill_value=0.0); X_test = (X_test - feature_means) / feature_stds

    model = RegLinealWine(learning_rate=args.learning_rate, num_iters=args.num_iters, l2_penalty=args.l2_penalty,
                      fit_intercept=not args.no_intercept, seed=args.seed, verbose=args.verbose)
    model.fit(X_train.to_numpy(float), y_train)
    preds = model.predict(X_test.to_numpy(float))
    print("Resultados en TEST:")
    print("  R2   =", round(r2(y_test, preds), 4))
    print("  MSE  =", round(np.mean((y_test - preds) ** 2), 4))
    print("  RMSE =", round(rmse(y_test, preds), 4))
    print("  MAE  =", round(mae(y_test, preds), 4))
    train_preds = model.predict(X_train.to_numpy(float))
    print("\nResultados en TRAIN:")
    print("  R2   =", round(r2(y_train, train_preds), 4))
    print("  MSE  =", round(np.mean((y_train - train_preds) ** 2), 4))
    print("  RMSE =", round(rmse(y_train, train_preds), 4))
    print("  MAE  =", round(mae(y_train, train_preds), 4))
    y_base = np.full_like(y_test, fill_value=float(y_train.mean()))
    print("\nBaseline (media de y_train) en TEST:")
    print("  R2   =", round(r2(y_test, y_base), 4))
    print("  MSE  =", round(np.mean((y_test - y_base) ** 2), 4))
    print("  RMSE =", round(rmse(y_test, y_base), 4))
    print("  MAE  =", round(mae(y_test, y_base), 4))

    model_info = dict(kind="wine_linreg", version=1, target=args.target, columns=features,
                feature_means=feature_means.astype(float).to_dict(), feature_stds=feature_stds.astype(float).to_dict(),
                fit_intercept=not args.no_intercept, learning_rate=args.learning_rate, num_iters=args.num_iters, l2_penalty=args.l2_penalty, seed=args.seed,
                w=model.w.ravel().tolist())
    save_json(args.model_path, model_info); print(f"\nmodelo guardado en model.json")

def predict_cmd(args):
    model_info = load_json(args.model_path)
    if model_info.get("kind") != "wine_linreg":
        print("error: modelo no reconocido")
        sys.exit(1)

    df = pd.read_csv(args.csv, sep=',').dropna().reset_index(drop=True)
    if model_info["target"] in df.columns:
        df = df.drop(columns=[model_info["target"]])
    df = df.reindex(columns=model_info["columns"], fill_value=0.0).astype(float)

    feature_means = pd.Series(model_info["feature_means"])
    feature_stds = pd.Series(model_info["feature_stds"])
    X = (df - feature_means) / feature_stds

    model = RegLinealWine(fit_intercept=bool(model_info["fit_intercept"]))
    model.w = np.asarray(model_info["w"], float).reshape(-1, 1)

    preds = model.predict(X.to_numpy(float))
    out = pd.DataFrame({"prediction": preds})

    print("primeras predicciones:")
    for val in out["prediction"].head(10):
        print(" ", round(val, 4))

    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\npredicciones guardadas en predicciones.csv")

def main():
    p = argparse.ArgumentParser(description="regresión lineal desde cero para el dataset Wine Quality")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="entrenar el modelo")
    t.add_argument("--csv", required=True); t.add_argument("--target", required=True)
    t.add_argument("--model-path", default="model.json")
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--learning_rate", type=float, default=0.05); t.add_argument("--num_iters", type=int, default=5000)
    t.add_argument("--l2_penalty", type=float, default=0.0); t.add_argument("--no-intercept", action="store_true")
    t.add_argument("--seed", type=int, default=42); t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=train_cmd)

    p2 = sub.add_parser("predict", help="predecir con un modelo entrenado")
    p2.add_argument("--csv", required=True); p2.add_argument("--model-path", default="model.json")
    p2.add_argument("--out-csv", default=None); p2.set_defaults(func=predict_cmd)

    args = p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
