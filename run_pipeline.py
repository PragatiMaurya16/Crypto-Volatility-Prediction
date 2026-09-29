"""One command to run everything: clean -> features -> EDA -> train -> reports.

    python run_pipeline.py            # full run on whatever is in data/raw/
    python run_pipeline.py --quick    # fast smoke test (less tuning)
"""
import argparse
import json

from src import config, data_loader, diagrams, eda, features, preprocessing, report_builder, style, train


def main(quick: bool = False):
    for d in (config.PROCESSED_DIR, config.MODEL_DIR, config.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading and cleaning data")
    raw = data_loader.load_raw()
    clean, report = preprocessing.clean(raw)
    clean.to_csv(config.PROCESSED_DIR / "clean_data.csv.gz", index=False)
    (config.REPORT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print("   ", report)

    print("2/5 Building features")
    feats = features.build_features(clean)
    feats.to_csv(config.PROCESSED_DIR / "features.csv.gz", index=False)
    print(f"    {feats.shape[0]:,} rows x {len(features.FEATURES)} features")

    print("3/5 Exploratory analysis")
    style.apply()
    eda.run(clean, feats, report)
    diagrams.pipeline()
    diagrams.architecture()

    print("4/5 Training models")
    train.main(quick=quick)

    print("5/5 Writing reports")
    report_builder.build()
    print("Done. Start the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
