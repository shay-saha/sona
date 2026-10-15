#!/usr/bin/env python3
"""Speaker-held-out pilot: age/height/weight from native C++ audio descriptors.

Hyperparameters and feature groups are chosen on training speakers only.
The official test speakers are used once for reporting, never for fitting.
"""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015
TARGETS = {
    "height_cm": ("Height", "cm"),
    "weight_kg": ("Weight", "kg"),
    "age_years": ("Age", "years"),
}
ALPHAS = [0.1, 1, 10, 100, 1000]


def score(y, p):
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
    }


def paired_improvement_interval(y, baseline, predictions):
    differences = np.abs(y - baseline) - np.abs(y - predictions)
    rng = np.random.default_rng(SEED)
    boot = rng.choice(differences, (5000, len(y)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def feature_groups(names):
    pitch = [n for n in names if n.startswith("pitch_") or n == "periodicity_mean"]
    spectrum = [
        n
        for n in names
        if n.startswith(("mfcc", "centroid_", "flatness_", "zcr_", "flux_"))
    ]
    return {
        "pitch": pitch,
        "spectral": spectrum,
        "combined": pitch
        + spectrum
        + ["harmonicity_median_db", "active_fraction", "periodic_fraction"],
        "recording_nuisance": [
            "duration_s",
            "level_mean_dbfs",
            "level_sd_db",
            "clipped_fraction",
        ],
    }


def correlations(frame, features):
    rows = []
    for target in TARGETS:
        for name in features:
            if frame[name].nunique() < 3:
                continue
            rho, p = spearmanr(frame[name], frame[target])
            rows.append(
                {
                    "target": target,
                    "feature": name,
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            )
    # Benjamini-Hochberg over every target/feature test, not just the interesting ones.
    order = np.argsort([r["p_value"] for r in rows])
    adjusted = np.array(
        [rows[i]["p_value"] * len(rows) / (rank + 1) for rank, i in enumerate(order)]
    )
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    for i, q in zip(order, adjusted):
        rows[int(i)]["fdr_q"] = float(min(1, q))
    return sorted(rows, key=lambda r: (r["fdr_q"], -abs(r["spearman_rho"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/nisp_manifest_Malayalam_master.json",
    )
    parser.add_argument(
        "--extractor", type=Path, default=ROOT.parent / "build/sona_features"
    )
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cache = ROOT / "cache"
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    feature_path = output / "nisp_pilot_features.csv"
    if not args.reuse_features or not feature_path.exists():
        paths = [record["path"] for record in manifest["records"]]
        print(f"Extracting {len(paths)} recordings using {args.extractor}", flush=True)
        with feature_path.open("w") as stream:
            subprocess.run(
                [str(args.extractor.resolve()), *paths],
                cwd=ROOT,
                stdout=stream,
                check=True,
            )
    clips = pd.read_csv(feature_path)
    clips["speaker_id"] = clips["file"].map(lambda p: Path(p).parent.name)
    names = [n for n in clips.columns if n not in ("file", "speaker_id")]
    if not np.isfinite(clips[names].values).all():
        raise ValueError("Non-finite features")
    metadata = pd.read_csv(cache / "nisp_total_spkrinfo.list", sep=r"\s+")
    labels = metadata[["Speaker_ID", "Height", "Weight", "Age"]].rename(
        columns={"Speaker_ID": "speaker_id", **{v[0]: k for k, v in TARGETS.items()}}
    )
    clips = clips.merge(labels, on="speaker_id", how="inner", validate="many_to_one")
    people = clips.groupby("speaker_id")[names + list(TARGETS)].mean()
    train_ids = {
        s.split("___")[0] for s in (cache / "nisp_train_spkrID").read_text().split()
    }
    test_ids = {
        s.split("___")[0] for s in (cache / "nisp_test_spkrID").read_text().split()
    }
    if train_ids & test_ids:
        raise ValueError("Official speaker splits overlap")
    train = people.loc[people.index.isin(train_ids)].copy()
    test = people.loc[people.index.isin(test_ids)].copy()
    if len(train) < 20 or len(test) < 10:
        raise ValueError("Insufficient independent speakers for the pilot")
    if set(train.index) & set(test.index):
        raise ValueError("Speaker leakage")
    groups = feature_groups(names)
    cv = list(KFold(n_splits=6, shuffle=True, random_state=SEED).split(train))
    report = {
        "dataset": "NISP",
        "source": "https://github.com/iiscleap/NISP-Dataset",
        "paper": "https://arxiv.org/abs/2007.06021",
        "revision": manifest["revision"],
        "group": manifest["group"],
        "recordings": len(clips),
        "speakers": len(people),
        "train_speakers": len(train),
        "test_speakers": len(test),
        "test_speaker_ids": list(test.index),
        "seed": SEED,
        "feature_schema": names,
        "protocol": "Official speaker-disjoint split; six utterances averaged per speaker; six-fold training-only CV selects alpha and feature group.",
        "versions": {
            n: importlib.metadata.version(n)
            for n in ["numpy", "scipy", "pandas", "scikit-learn"]
        },
        "python": platform.python_version(),
        "targets": {},
    }
    prediction_rows = []
    native_models = []
    print(
        f"{len(train)} training speakers; {len(test)} official unseen test speakers; {len(clips)} clips",
        flush=True,
    )
    with threadpool_limits(limits=1):
        for target, (_, unit) in TARGETS.items():
            y = train[target].to_numpy()
            yt = test[target].to_numpy()
            baseline_cv = np.zeros(len(train))
            for fit, val in cv:
                baseline_cv[val] = np.median(y[fit])
            baseline_value = float(np.median(y))
            baseline = np.full(len(test), baseline_value)
            result = {
                "unit": unit,
                "train_label_range": [float(y.min()), float(y.max())],
                "constant_baseline": score(yt, baseline),
                "constant_value": baseline_value,
                "training_cv_baseline_mae": float(mean_absolute_error(y, baseline_cv)),
                "models": {},
            }
            fitted = {}
            for group, columns in groups.items():
                candidates = []
                x = train[columns].to_numpy()
                for alpha in ALPHAS:
                    oof = np.zeros(len(train))
                    for fit, val in cv:
                        pipeline = make_pipeline(
                            StandardScaler(), Ridge(alpha=alpha, solver="svd")
                        )
                        pipeline.fit(x[fit], y[fit])
                        oof[val] = pipeline.predict(x[val])
                    candidates.append((float(mean_absolute_error(y, oof)), alpha, oof))
                cv_error, alpha, oof = min(candidates, key=lambda r: r[0])
                pipeline = make_pipeline(
                    StandardScaler(), Ridge(alpha=alpha, solver="svd")
                ).fit(x, y)
                pred = pipeline.predict(test[columns].to_numpy())
                result["models"][group] = {
                    **score(yt, pred),
                    "training_cv_mae": cv_error,
                    "alpha": alpha,
                    "paired_improvement_95ci": paired_improvement_interval(
                        yt, baseline, pred
                    ),
                }
                fitted[group] = (pipeline, pred, columns)
            # This is selected solely by training CV, without looking at test errors.
            selected = min(
                result["models"], key=lambda k: result["models"][k]["training_cv_mae"]
            )
            model, predictions, columns = fitted[selected]
            result["selected_by_training_cv"] = selected
            result["selected_result"] = result["models"][selected]
            # Predeclared research gate; a positive point estimate alone is insufficient.
            s = result["selected_result"]
            result["passes_pilot_gate"] = bool(
                selected != "recording_nuisance"
                and s["training_cv_mae"] < 0.9 * result["training_cv_baseline_mae"]
                and s["paired_improvement_95ci"][0] > 0
            )
            for person, truth, prediction in zip(test.index, yt, predictions):
                prediction_rows.append(
                    {
                        "speaker_id": person,
                        "target": target,
                        "truth": float(truth),
                        "prediction": float(prediction),
                        "constant_baseline": baseline_value,
                    }
                )
            scaler, ridge = model.steps[0][1], model.steps[1][1]
            weights = np.zeros(len(names))
            means = np.zeros(len(names))
            scales = np.ones(len(names))
            for i, name in enumerate(columns):
                index = names.index(name)
                weights[index] = ridge.coef_[i]
                means[index] = scaler.mean_[i]
                scales[index] = scaler.scale_[i]
            native_models.append(
                {
                    "target": target,
                    "unit": unit,
                    "experimental": True,
                    "passes_pilot_gate": result["passes_pilot_gate"],
                    "feature_names": names,
                    "means": means.tolist(),
                    "scales": scales.tolist(),
                    "weights": weights.tolist(),
                    "intercept": float(ridge.intercept_),
                    "validation": s,
                    "trained_on": f"NISP {manifest['group']}, {len(train)} speakers; average of six clips",
                }
            )
            # Same unaltered fitted model, fewer test utterances: evidence accumulation test.
            result["utterance_accumulation"] = {}
            test_clips = clips[clips.speaker_id.isin(test.index)]
            for count in [1, 3, 6]:
                subset = (
                    test_clips.groupby("speaker_id", sort=False)
                    .head(count)
                    .groupby("speaker_id")[columns]
                    .mean()
                    .reindex(test.index)
                )
                result["utterance_accumulation"][str(count)] = score(
                    yt, model.predict(subset.to_numpy())
                )
            # Explicitly invalid split diagnostic: speaker memorization can inflate apparent accuracy.
            diagnostic_columns = groups["combined"]
            for split_name in ["random_clips_invalid", "official_unseen_speakers"]:
                if split_name == "random_clips_invalid":
                    fit_clips, eval_clips = train_test_split(
                        clips, test_size=0.3, random_state=SEED
                    )
                else:
                    fit_clips = clips[clips.speaker_id.isin(train.index)]
                    eval_clips = test_clips
                knn = make_pipeline(
                    StandardScaler(),
                    KNeighborsRegressor(n_neighbors=3, weights="distance"),
                )
                knn.fit(
                    fit_clips[diagnostic_columns].to_numpy(),
                    fit_clips[target].to_numpy(),
                )
                temp = eval_clips[["speaker_id", target]].copy()
                temp["prediction"] = knn.predict(
                    eval_clips[diagnostic_columns].to_numpy()
                )
                aggregate = temp.groupby("speaker_id")[[target, "prediction"]].mean()
                result[split_name + "_diagnostic"] = {
                    **score(aggregate[target], aggregate.prediction),
                    "speaker_overlap": len(
                        set(fit_clips.speaker_id) & set(eval_clips.speaker_id)
                    ),
                }
            report["targets"][target] = result
            print(
                f"{target}: baseline MAE {result['constant_baseline']['mae']:.3f}; {selected} MAE {s['mae']:.3f} {unit}; paired improvement CI {s['paired_improvement_95ci']}; gate {result['passes_pilot_gate']}",
                flush=True,
            )
    report["training_only_correlations"] = correlations(train, names)
    report["limitations"] = [
        "Pilot: one language cohort, read English, only 60 people; not representative ambient audio.",
        "Official test speakers never occur in training; clip/window counts do not increase independent participant count.",
        "Age labels have a narrow adult range in this cohort.",
        "Height, weight and age are the only targets. Protected-attribute and health labels are not inputs or targets.",
        "Train-only correlations are exploratory associations, not causal physical mechanisms.",
        "No independent corpus or real-room validation yet. A pilot gate is not a deployment approval.",
    ]
    (output / "nisp_pilot_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    (output / "nisp_pilot_models.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "extractor": "sona_features_v1",
                "provenance": {
                    "dataset": "NISP",
                    "source": "https://github.com/iiscleap/NISP-Dataset",
                    "paper": "https://arxiv.org/abs/2007.06021",
                    "revision": manifest["revision"],
                    "licence": "CC-BY-4.0 per source repository; paper describes academic/research use",
                    "seed": SEED,
                },
                "models": native_models,
            },
            indent=2,
            allow_nan=False,
        )
    )
    pd.DataFrame(prediction_rows).to_csv(
        output / "nisp_pilot_predictions.csv", index=False
    )
    print("Saved report, predictions, and portable model parameters.", flush=True)


if __name__ == "__main__":
    main()
