"""Train, tune, compare and save the volatility model.

Run:  python -m src.train           (full)
      python -m src.train --quick   (fewer tuning iterations, for a fast test)
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from . import config, data_loader, evaluate, features as F, modeling


def load_features() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "features.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` first (features not built).")
    return pd.read_csv(path, parse_dates=["date"])


def main(quick: bool = False) -> dict:
    t0 = time.time()
    df = load_features()
    df = df.dropna(subset=[F.TARGET]).sort_values(["date", "symbol"]).reset_index(drop=True)
    train, val, test, split_info = modeling.chronological_split(df)
    print(f"Rows  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    X_tr, y_tr = train[F.FEATURES], train[F.TARGET]
    X_va, y_va = val[F.FEATURES], val[F.TARGET]
    X_te, y_te = test[F.FEATURES], test[F.TARGET]
    n_coins = df["symbol"].nunique()

    # ---------- 1. tune each candidate with forward-chaining CV on TRAIN ----------
    tuned, val_scores, best_params, cv_scores = {}, {}, {}, {}
    for name in modeling.candidates():
        s = modeling.tune(name, n_coins, n_iter=2 if quick else modeling.N_ITER[name], n_splits=3 if quick else 4)
        s.fit(X_tr, y_tr)
        i = int(np.argmax(s.cv_results_["mean_test_score"]))
        params = s.cv_results_["params"][i]
        cv_scores[name] = float(-s.cv_results_["mean_test_score"][i])
        est, _ = modeling.candidates()[name]
        model = modeling.make_pipeline(clone(est)).set_params(**params).fit(X_tr, y_tr)
        val_scores[name] = evaluate.regression_metrics(y_va, model.predict(X_va))
        best_params[name] = {k.split("__")[-1]: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                             for k, v in params.items()}
        tuned[name] = params
        print(f"  {name:<18} CV-RMSE={cv_scores[name]:.5f}  VAL-RMSE={val_scores[name]['RMSE']:.5f}")

    val_scores["Naive (last 7 days)"] = evaluate.regression_metrics(y_va, X_va[F.BASELINE])
    best_name = min((k for k in tuned), key=lambda k: val_scores[k]["RMSE"])
    print(f"Selected on validation: {best_name}")

    # ---------- 2. refit every candidate on TRAIN+VAL, score once on TEST ----------
    full = pd.concat([train, val])
    X_full, y_full = full[F.FEATURES], full[F.TARGET]
    final, test_scores, preds = {}, {}, {}
    for name, params in tuned.items():
        est, _ = modeling.candidates()[name]
        m = modeling.make_pipeline(clone(est)).set_params(**params).fit(X_full, y_full)
        final[name] = m
        preds[name] = m.predict(X_te)
        test_scores[name] = evaluate.regression_metrics(y_te, preds[name])
    test_scores["Naive (last 7 days)"] = evaluate.regression_metrics(y_te, X_te[F.BASELINE])
    table = pd.DataFrame(test_scores).T
    print("\nTest set:\n", table.round(5).to_string())

    best = final[best_name]
    p = preds[best_name]

    # ---------- 3. regimes, importance, plots ----------
    thresholds = [float(np.quantile(y_full, q)) for q in config.REGIME_QUANTILES]
    regime = evaluate.regime_report(y_te, p, thresholds)
    imp = evaluate.feature_importance(best, X_te, y_te)
    evaluate.plot_model_comparison(table)
    evaluate.plot_pred_vs_actual(y_te.values, p)
    evaluate.plot_residuals(y_te.values, p)
    sample_coin = evaluate.plot_timeseries(test, p)
    evaluate.plot_confusion(regime["confusion"])
    evaluate.plot_importance(imp)

    # per-coin accuracy on the test set
    per_coin = (test.assign(pred=p).groupby("symbol")
                .apply(lambda g: pd.Series(evaluate.regression_metrics(g[F.TARGET], g["pred"])), include_groups=False)
                .sort_values("RMSE"))
    per_coin.to_csv(config.REPORT_DIR / "per_coin_metrics.csv")

    # ---------- 4. save ----------
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, config.MODEL_DIR / "volatility_model.joblib", compress=3)
    test.assign(predicted_vol=p)[["date", "symbol", F.TARGET, "predicted_vol", "vol_7"]].to_csv(
        config.REPORT_DIR / "test_predictions.csv", index=False)
    imp.to_csv(config.REPORT_DIR / "feature_importance.csv", index=False)

    meta = {
        "model_name": best_name,
        "best_params": best_params,
        "features": F.FEATURES,
        "horizon_days": config.HORIZON,
        "regime_thresholds": thresholds,
        "regime_labels": list(config.REGIME_LABELS),
        "split": split_info,
        "cv_rmse": cv_scores,
        "validation_metrics": val_scores,
        "test_metrics": test_scores,
        "regime_test": regime,
        "sample_coin": sample_coin,
        "n_coins": int(n_coins),
        "is_demo": data_loader.is_demo_data(),
        "n_features": len(F.FEATURES),
        "top_features": imp.head(10)["feature"].tolist(),
        "train_seconds": round(time.time() - t0, 1),
    }
    (config.MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + metadata in {time.time()-t0:.0f}s")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
