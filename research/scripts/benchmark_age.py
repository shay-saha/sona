#!/usr/bin/env python3
"""Adult age calibration with separate fit, interval-calibration and test actors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from age_inference import AgeEncoder
from benchmark_physiology import groups
from experiment_utils import (
    ROOT,
    SEED,
    conformal_radius,
    evaluate,
    extract_features,
    mae,
    perturb_records,
    save_json,
)


def predict_records(encoder, records, destination, reuse):
    if reuse and destination.exists():
        table = pd.read_csv(destination)
        if table.file.tolist() == [r["path"] for r in records]:
            return table
    rows = []
    for i, record in enumerate(records):
        rows.append(
            {
                "file": record["path"],
                "speaker_id": str(record["speaker_id"]),
                "raw_age": encoder.predict(ROOT / record["path"]),
            }
        )
        if (i + 1) % 24 == 0:
            print(f"Age-only encoder: {i + 1}/{len(records)}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(destination, index=False)
    return table


def calibrate(kind, x, y):
    if kind == "raw":
        return 1.0, 0.0
    if kind == "training_mean_bias":
        return 1.0, float(np.mean(y - x))
    model = Ridge(alpha=float(kind.rsplit("_", 1)[1])).fit(x[:, None], y)
    return float(model.coef_[0]), float(model.intercept_)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--extractor", type=Path, default=ROOT.parent / "build/sona_features"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "results/crema_manifest.json").read_text())
    labels = pd.DataFrame(manifest["people"]).set_index("speaker_id").sort_index()
    ids = labels.index.to_numpy()
    brackets = (labels.age_years // 15).clip(upper=4)
    remaining, test_ids = train_test_split(
        ids, test_size=18, random_state=SEED, stratify=brackets
    )
    fit_ids, cal_ids = train_test_split(
        remaining, test_size=18, random_state=SEED, stratify=brackets.loc[remaining]
    )
    split = {
        "fit_ids": fit_ids.tolist(),
        "calibration_ids": cal_ids.tolist(),
        "test_ids": test_ids.tolist(),
    }
    # Persist identities before running any graph or selecting calibrators.
    save_json(ROOT / "results/crema_age_split.json", {"seed": SEED, **split})
    records = manifest["records"]
    model_path = ROOT / "cache/age_encoder/age_only.onnx"
    encoder = AgeEncoder(model_path)
    clips = predict_records(
        encoder, records, ROOT / "results/crema_raw_age.csv", args.reuse_features
    )
    clips.speaker_id = clips.speaker_id.astype(str)
    raw = clips.groupby("speaker_id").raw_age.mean().loc[ids]
    y = labels.loc[fit_ids].age_years.to_numpy(dtype=float)
    x = raw.loc[fit_ids].to_numpy()
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(y))
    cv_baseline = np.zeros(len(y))
    for train, val in folds:
        cv_baseline[val] = np.median(y[train])
    rows = []
    for kind in [
        "raw",
        "training_mean_bias",
        "affine_ridge_0.1",
        "affine_ridge_1",
        "affine_ridge_10",
    ]:
        oof = np.zeros(len(y))
        for train, val in folds:
            slope, intercept = calibrate(kind, x[train], y[train])
            oof[val] = x[val] * slope + intercept
        rows.append({"kind": kind, "cv_mae": mae(y, oof)})
    selected = min(rows, key=lambda row: row["cv_mae"])
    slope, intercept = calibrate(selected["kind"], x, y)
    first = clips.groupby("speaker_id").head(1).set_index("speaker_id").raw_age
    radius = conformal_radius(
        labels.loc[cal_ids].age_years.to_numpy(),
        first.loc[cal_ids].to_numpy() * slope + intercept,
    )
    truth = labels.loc[test_ids].age_years.to_numpy()
    baseline = np.full(len(test_ids), np.median(y))
    predictions = raw.loc[test_ids].to_numpy() * slope + intercept
    report = {
        "source": manifest["source"],
        "revision": manifest["revision"],
        "seed": SEED,
        **split,
        "selection": selected,
        "candidates": rows,
        "cv_baseline_mae": mae(y, cv_baseline),
        "test": evaluate(truth, predictions, baseline, radius),
        "stress": {},
        "limitations": [
            "Performed neutral speech, 91 actors, ages 20–74; not an ambient validation.",
            "Pretraining identities unavailable; actor overlap with source corpora cannot be excluded.",
            "Six-utterance speaker averages; single-window accuracy reported separately.",
        ],
    }
    for count in [1, 3, 6]:
        accumulated = (
            clips.groupby("speaker_id")
            .head(count)
            .groupby("speaker_id")
            .raw_age.mean()
            .loc[test_ids]
        )
        p = accumulated.to_numpy() * slope + intercept
        report.setdefault("utterance_accumulation", {})[str(count)] = {
            "mae": mae(truth, p),
            "interval_coverage": float(np.mean(np.abs(truth - p) <= radius)),
        }
    # A matched native-descriptor baseline, selected entirely on fit actors.
    descriptors = extract_features(
        records,
        args.extractor,
        ROOT / "results/crema_features.csv",
        args.reuse_features,
    )
    names = descriptors.columns[1:].tolist()
    descriptors["speaker_id"] = [r["speaker_id"] for r in records]
    people = descriptors.groupby("speaker_id")[names].mean()
    native_rows = []
    with threadpool_limits(limits=1):
        for group, columns in groups(names).items():
            for alpha in [1, 10, 100, 1000]:
                features = people.loc[fit_ids, columns].to_numpy()
                oof = np.zeros(len(y))
                for train, val in folds:
                    model = make_pipeline(
                        StandardScaler(), Ridge(alpha=alpha, solver="svd")
                    ).fit(features[train], y[train])
                    oof[val] = model.predict(features[val])
                native_rows.append(
                    {
                        "group": group,
                        "columns": columns,
                        "alpha": alpha,
                        "cv_mae": mae(y, oof),
                    }
                )
        best_native = min(native_rows, key=lambda row: row["cv_mae"])
        cols = best_native["columns"]
        native = make_pipeline(
            StandardScaler(), Ridge(alpha=best_native["alpha"], solver="svd")
        ).fit(people.loc[fit_ids, cols].to_numpy(), y)
        report["native_baseline"] = {
            "selected": best_native,
            "test_mae": mae(
                truth, native.predict(people.loc[test_ids, cols].to_numpy())
            ),
            "candidates": native_rows,
        }
    test_records = [r for r in records if r["speaker_id"] in test_ids]
    for condition in ["gain_0.25", "white_noise_15db"]:
        altered = perturb_records(
            test_records, condition, ROOT / f"cache/crema_age/{condition}"
        )
        table = predict_records(
            encoder,
            altered,
            ROOT / f"results/crema_age_{condition}.csv",
            args.reuse_features,
        )
        table.speaker_id = table.speaker_id.astype(str)
        p = (
            table.groupby("speaker_id").raw_age.mean().loc[test_ids].to_numpy() * slope
            + intercept
        )
        report["stress"][condition] = {"mae": mae(truth, p)}
    passed = (
        selected["cv_mae"] < 0.9 * report["cv_baseline_mae"]
        and report["test"]["paired_improvement_95ci"][0] > 0
        and all(
            s["mae"] <= report["test"]["mae"] * 1.2 for s in report["stress"].values()
        )
    )
    report["passes_gate"] = bool(passed)
    save_json(ROOT / "results/crema_age_report.json", report)
    provenance = json.loads((ROOT / "cache/age_encoder/provenance.json").read_text())
    calibration = {
        "schema_version": 1,
        "target": "age_years",
        "unit": "years",
        "input": encoder.input,
        "output": encoder.output,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "slope": slope,
        "intercept": intercept,
        "interval_radius": radius,
        "validation_mae": report["utterance_accumulation"]["1"]["mae"],
        "passes_gate": bool(passed),
        "scope": "CREMA-D neutral adult speech, ages 20–74; 55 fit / 18 calibration / 18 test actors",
        "provenance": provenance,
    }
    save_json(ROOT / "results/age_calibration.json", calibration)
    pd.DataFrame(
        {
            "speaker_id": test_ids,
            "truth": truth,
            "prediction": predictions,
            "interval_lower": predictions - radius,
            "interval_upper": predictions + radius,
        }
    ).to_csv(ROOT / "results/crema_age_predictions.csv", index=False)
    print(
        json.dumps(
            {
                "selected": selected,
                "test": report["test"],
                "stress": report["stress"],
                "passes_gate": passed,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
