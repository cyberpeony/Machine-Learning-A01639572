"""
Regresión Lineal desde cero aplicada al dataset Wine Quality
- Optimización con gradiente descendente (regularización l2 penalty)
- Preprocesamiento con estandarización (fit solo con train)

Training simple:
    python src/regLinealWine.py train --csv data/winequality-red.csv --target quality --model-path model.json --learning_rate 0.05 --num_iters 6000 --l2_penalty 0.0
Predicción:
    python src/regLinealWine.py predict --csv data/winequality-red.csv --model-path model.json --out-csv predicciones.csv
Validación cruzada de 5 folds:
    python src/regLinealWine.py train --csv data/winequality-red.csv --target quality \
      --model-path model.json --num_iters 6000 --cv 5
Grid search en hiperparámetros (lr y l2):
    python src/regLinealWine.py train --csv data/winequality-red.csv --target quality \
      --model-path model.json --num_iters 6000 --cv 5 \
      --grid_lr 0.01 0.05 0.1 --grid_l2 0.0 0.01 0.1
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

def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idxs = np.arange(n); rng.shuffle(idxs)
    folds = np.array_split(idxs, k)
    splits = []
    for fold in range(k):
        val_indices = folds[fold]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != fold])
        splits.append((train_indices, val_indices))
    return splits

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
    
    # validación cruzada k-folds 
    if args.cv and args.cv >= 2:
        k_folds = args.cv
        folds = kfold_indices(len(df), k=k_folds, seed=args.seed)
        lr_grid = args.grid_lr if args.grid_lr else [args.learning_rate]
        l2_grid = args.grid_l2 if args.grid_l2 else [args.l2_penalty]
        crossval_resultados = []  # (lr, l2, r2_mean, r2_std, rmse_mean, mae_mean)
        for lr in lr_grid:
            for l2 in l2_grid:
                r2_scores_fold, rmse_scores, mae_scores = [], [], []
                for train_indices, val_indices in folds:
                    X_train_fold = X.iloc[train_indices].copy()
                    X_val_fold   = X.iloc[val_indices].copy()
                    y_train_fold = y[train_indices]
                    y_val_fold   = y[val_indices]
                    # estandarizar con estadísticas del fold-train
                    feature_names = X_train_fold.columns.tolist()
                    mean_train = X_train_fold.mean(0)
                    std_train  = X_train_fold.std(0, ddof=0).replace(0.0, 1.0)
                    X_train_fold = (X_train_fold - mean_train) / std_train
                    X_val_fold   = X_val_fold.reindex(columns=feature_names, fill_value=0.0)
                    X_val_fold   = (X_val_fold - mean_train) / std_train
                    model_crossval = RegLinealWine(
                        learning_rate=lr, num_iters=args.num_iters, l2_penalty=l2,
                        fit_intercept=not args.no_intercept, seed=args.seed, verbose=False
                    )
                    model_crossval.fit(X_train_fold.to_numpy(float), y_train_fold)
                    y_val_pred = model_crossval.predict(X_val_fold.to_numpy(float))
                    r2_scores_fold.append(r2(y_val_fold, y_val_pred))
                    rmse_scores.append(rmse(y_val_fold, y_val_pred))
                    mae_scores.append(mae(y_val_fold, y_val_pred))
                crossval_resultados.append((
                    lr, l2,
                    float(np.mean(r2_scores_fold)), float(np.std(r2_scores_fold)),
                    float(np.mean(rmse_scores)),    float(np.mean(mae_scores))
                ))
        # resumen de la crossval
        print(f"Resultados {k_folds}-fold crossval (promedio +- desv. estándar)")
        for lr, l2, r2_mean, r2_std, rmse_mean, mae_mean in crossval_resultados:
            print(f"lr={lr:g}  l2={l2:g}  R2={r2_mean:.4f}+-{r2_std:.4f}  RMSE={rmse_mean:.4f}  MAE={mae_mean:.4f}")
        # mejores hiperparámetros por R2 promedio
        best_hiperparms = max(crossval_resultados, key=lambda t: (t[2], -t[4])) 
        best_lr, best_l2 = best_hiperparms[0], best_hiperparms[1]
        print(f"\nMejores hiperparámetros por R2 promedio: lr={best_lr:g}, l2={best_l2:g}")
        print(f"Hiperparámetros finales: lr={best_lr:g}, l2={best_l2:g}, iters={args.num_iters}, seed={args.seed}")
        # final training (con todo el dataset)
        feature_names_all = X.columns.tolist()
        means_all = X.mean(0)
        stds_all  = X.std(0, ddof=0).replace(0.0, 1.0)
        X_all_std = (X - means_all) / stds_all
        final_model = RegLinealWine(
            learning_rate=best_lr, num_iters=args.num_iters, l2_penalty=best_l2,
            fit_intercept=not args.no_intercept, seed=args.seed, verbose=args.verbose
        )
        final_model.fit(X_all_std.to_numpy(float), y)
        model_info = dict(
            kind="wine_linreg", version=1, target=args.target, columns=feature_names_all,
            feature_means=means_all.astype(float).to_dict(), feature_stds=stds_all.astype(float).to_dict(),
            fit_intercept=not args.no_intercept, learning_rate=best_lr, num_iters=args.num_iters,
            l2_penalty=best_l2, seed=args.seed, w=final_model.w.ravel().tolist()
        )
        save_json(args.model_path, model_info)
        print(f"\nmodelo final (entrenado con todo el dataset) guardado en {args.model_path}")
        return

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
    t.add_argument("--cv", type=int, default=0, help="k-fold CV >=2 para evaluar")
    t.add_argument("--grid_lr", nargs="*", type=float, default=None, help="lista learning_rates p/grid")
    t.add_argument("--grid_l2", nargs="*", type=float, default=None, help="lista l2_penalty p/grid")
    t.set_defaults(func=train_cmd)

    p2 = sub.add_parser("predict", help="predecir con un modelo entrenado")
    p2.add_argument("--csv", required=True); p2.add_argument("--model-path", default="model.json")
    p2.add_argument("--out-csv", default=None); p2.set_defaults(func=predict_cmd)

    args = p.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
