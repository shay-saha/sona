#!/usr/bin/env python3
"""Frozen, cohort-disjoint model selection followed by a fresh-speaker audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

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

TARGETS = {
    "height_cm": ("Height", "cm"),
    "weight_kg": ("Weight", "kg"),
    "age_years": ("Age", "years"),
    "shoulder_cm": ("Shoulder_size", "cm"),
}
COHORTS = ["Hindi", "Kannada", "Malayalam", "Tamil", "Telugu"]


def groups(names):
    pitch = [n for n in names if n.startswith("pitch_") or n == "periodicity_mean"]
    spectral = [
        n
        for n in names
        if n.startswith(("mfcc", "centroid_", "flatness_", "zcr_", "flux_"))
    ]
    resonance = [
        n for n in names if n.startswith(("f1_", "f2_", "f3_", "f4_", "formant_"))
    ]
    voiced = [n for n in names if n.startswith("voiced_")]
    return {
        "pitch": pitch,
        "legacy_spectral": spectral,
        "resonance": resonance,
        "voiced_spectral": voiced,
        "resonance_and_voiced": resonance + voiced + pitch,
        "combined": pitch + spectral + resonance + voiced + ["harmonicity_median_db"],
        "recording_nuisance": [
            "duration_s",
            "level_mean_dbfs",
            "level_sd_db",
            "clipped_fraction",
        ],
    }


def candidates():
    for alpha in [1, 10, 100, 1000]:
        yield {"kind": "ridge", "alpha": alpha}
    for c in [10, 100]:
        for gamma in [1, 3]:
            yield {"kind": "rbf_svr", "C": c, "gamma_factor": gamma, "epsilon": 0.1}


def pipeline(config, count):
    estimator = (
        Ridge(alpha=config["alpha"], solver="svd")
        if config["kind"] == "ridge"
        else SVR(
            C=config["C"],
            gamma=config["gamma_factor"] / count,
            epsilon=config["epsilon"],
        )
    )
    return make_pipeline(StandardScaler(), estimator)


def export_model(model, columns, names):
    scaler, estimator = (step[1] for step in model.steps)
    indices = [names.index(n) for n in columns]
    means, scales = np.zeros(len(names)), np.ones(len(names))
    means[indices], scales[indices] = scaler.mean_, scaler.scale_
    result = {
        "feature_names": names,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "intercept": float(np.asarray(estimator.intercept_).ravel()[0]),
    }
    if isinstance(estimator, Ridge):
        weights = np.zeros(len(names))
        weights[indices] = estimator.coef_
        result.update(kind="ridge", weights=weights.tolist())
    else:
        result.update(
            kind="rbf_svr",
            feature_indices=indices,
            support_vectors=estimator.support_vectors_.tolist(),
            dual_coefficients=estimator.dual_coef_[0].tolist(),
            gamma=float(estimator._gamma),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--extractor",
        type=Path,
        default=ROOT.parent / "build/sona_features",
    )
    args = parser.parse_args()
    records = []
    for cohort in COHORTS:
        manifest = json.loads(
            (ROOT / f"results/nisp_manifest_{cohort}_master.json").read_text()
        )
        records.extend(manifest["records"])
    clips = extract_features(
        records,
        args.extractor,
        ROOT / "results/nisp_expanded_features.csv",
        args.reuse_features,
    )
    names = clips.columns[1:].tolist()
    clips["speaker_id"] = [r["speaker_id"] for r in records]
    people = clips.groupby("speaker_id")[names].mean()
    metadata = pd.read_csv(
        ROOT / "cache/nisp_total_spkrinfo.list", sep=r"\s+"
    ).set_index("Speaker_ID")
    for target, (column, _) in TARGETS.items():
        people[target] = pd.to_numeric(metadata.loc[people.index, column])
    train_ids = {
        s.split("___")[0]
        for s in (ROOT / "cache/nisp_train_spkrID").read_text().split()
    }
    test_ids = {
        s.split("___")[0] for s in (ROOT / "cache/nisp_test_spkrID").read_text().split()
    }
    train = people.loc[people.index.isin(train_ids)]
    test = people.loc[people.index.isin(test_ids)]
    assert (
        len(people) == 345
        and len(train) == 210
        and len(test) == 135
        and not train_ids & test_ids
    )
    fit_ids, cal_ids = train_test_split(
        train.index,
        test_size=0.2,
        random_state=SEED,
        stratify=[i.split("_")[0] for i in train.index],
    )
    fit, cal = people.loc[fit_ids], people.loc[cal_ids]
    old_test = json.loads((ROOT / "results/nisp_pilot_report.json").read_text())[
        "test_speaker_ids"
    ]
    fresh = test.loc[~test.index.isin(old_test)]
    assert len(fresh) == 111
    folds = list(GroupKFold(5).split(fit, groups=[i.split("_")[0] for i in fit.index]))
    report = {
        "experiment": "physiology_v2",
        "seed": SEED,
        "recordings": len(clips),
        "fit_ids": fit_ids.tolist(),
        "calibration_ids": cal_ids.tolist(),
        "fresh_test_ids": fresh.index.tolist(),
        "legacy_test_ids": old_test,
        "protocol": json.loads((ROOT / "experiments/physiology_v2.json").read_text()),
        "targets": {},
    }
    selected_models, exported, predictions = {}, [], []
    with threadpool_limits(limits=1):
        for target, (_, unit) in TARGETS.items():
            y = fit[target].to_numpy()
            baseline_oof = np.zeros(len(y))
            for train_i, val_i in folds:
                baseline_oof[val_i] = np.median(y[train_i])
            baseline_cv = mae(y, baseline_oof)
            rows = []
            for group, columns in groups(names).items():
                x = fit[columns].to_numpy()
                for config in candidates():
                    oof = np.zeros(len(y))
                    for train_i, val_i in folds:
                        model = pipeline(config, len(columns)).fit(
                            x[train_i], y[train_i]
                        )
                        oof[val_i] = model.predict(x[val_i])
                    rows.append(
                        {
                            "group": group,
                            "columns": columns,
                            "config": config,
                            "cv_mae": mae(y, oof),
                        }
                    )
            # Select using cohort-disjoint training folds before any test predictions.
            best = min(
                (r for r in rows if r["group"] != "recording_nuisance"),
                key=lambda r: r["cv_mae"],
            )
            columns = best["columns"]
            model = pipeline(best["config"], len(columns)).fit(
                fit[columns].to_numpy(), y
            )
            selected_models[target] = (model, columns)
            # The interface consumes one window. Calibrate on one utterance per identity,
            # rather than claiming that six-utterance-average coverage applies to it.
            calibration_clips = (
                clips.loc[clips.speaker_id.isin(cal.index)]
                .groupby("speaker_id")
                .head(1)
                .set_index("speaker_id")
                .loc[cal.index]
            )
            radius = conformal_radius(
                cal[target].to_numpy(),
                model.predict(calibration_clips[columns].to_numpy()),
            )
            baseline = float(np.median(y))
            result = {
                "unit": unit,
                "selected": best,
                "cv_baseline_mae": baseline_cv,
                "label_range": [float(y.min()), float(y.max())],
                "candidates": rows,
                "fresh_test": evaluate(
                    fresh[target].to_numpy(),
                    model.predict(fresh[columns].to_numpy()),
                    np.full(len(fresh), baseline),
                    radius,
                ),
                "legacy_test": evaluate(
                    test.loc[old_test, target].to_numpy(),
                    model.predict(test.loc[old_test, columns].to_numpy()),
                    np.full(len(old_test), baseline),
                    radius,
                ),
                "stress": {},
            }
            result["cohort_audit"] = {}
            for cohort in sorted({i.split("_")[0] for i in fresh.index}):
                subset = fresh.loc[[i.split("_")[0] == cohort for i in fresh.index]]
                result["cohort_audit"][cohort] = evaluate(
                    subset[target].to_numpy(),
                    model.predict(subset[columns].to_numpy()),
                    np.full(len(subset), baseline),
                    radius,
                )
            # Provided labels enter this aggregate audit only, never any audio model.
            result["provided_metadata_confound_control"] = {}
            for label in sorted(metadata.Gender.unique()):
                fit_group = fit.loc[metadata.loc[fit.index, "Gender"] == label]
                test_group = fresh.loc[metadata.loc[fresh.index, "Gender"] == label]
                if len(fit_group) and len(test_group):
                    result["provided_metadata_confound_control"][label] = {
                        "speakers": len(test_group),
                        "audio_mae": mae(
                            test_group[target],
                            model.predict(test_group[columns].to_numpy()),
                        ),
                        "oracle_metadata_median_mae": mae(
                            test_group[target],
                            np.full(len(test_group), fit_group[target].median()),
                        ),
                    }
            result["utterance_accumulation"] = {}
            fresh_clips = clips.loc[clips.speaker_id.isin(fresh.index)]
            for count in [1, 3, 6]:
                subset = (
                    fresh_clips.groupby("speaker_id")
                    .head(count)
                    .groupby("speaker_id")[columns]
                    .mean()
                    .loc[fresh.index]
                )
                result["utterance_accumulation"][str(count)] = {
                    "mae": mae(fresh[target], model.predict(subset.to_numpy())),
                    "interval_coverage": float(
                        np.mean(
                            np.abs(
                                fresh[target].to_numpy()
                                - model.predict(subset.to_numpy())
                            )
                            <= radius
                        )
                    ),
                }
            result["fresh_test"]["single_utterance_mae"] = result[
                "utterance_accumulation"
            ]["1"]["mae"]
            report["targets"][target] = result
            for speaker, pred in zip(
                test.index, model.predict(test[columns].to_numpy())
            ):
                predictions.append(
                    {
                        "speaker_id": speaker,
                        "target": target,
                        "truth": float(test.loc[speaker, target]),
                        "prediction": float(pred),
                        "fresh": speaker not in old_test,
                    }
                )
            exported.append(
                {
                    **export_model(model, columns, names),
                    "target": target,
                    "unit": unit,
                    "passes_pilot_gate": False,
                    "interval_radius": radius,
                    "trained_on": "NISP five English cohorts; 168 fit / 42 calibration / 111 fresh-test speakers",
                    "validation": result["fresh_test"],
                }
            )
            print(
                target, best["group"], best["config"], result["fresh_test"], flush=True
            )

        # A frozen deployed control: never refit or select from its test scores.
        old_height = json.loads((ROOT / "results/nisp_pilot_models.json").read_text())[
            "models"
        ][0]

        def frozen_height(frame):
            x = frame[old_height["feature_names"]].to_numpy()
            return ((x - old_height["means"]) / old_height["scales"]) @ np.asarray(
                old_height["weights"]
            ) + old_height["intercept"]

        report["frozen_height_control"] = {
            "fresh_mae": mae(fresh.height_cm, frozen_height(fresh)),
            "stress": {},
        }
        from experiment_utils import paired_interval

        report["frozen_height_control"]["new_model_improvement_95ci"] = paired_interval(
            fresh.height_cm.to_numpy(),
            frozen_height(fresh),
            selected_models["height_cm"][0].predict(
                fresh[selected_models["height_cm"][1]].to_numpy()
            ),
        )
        fresh_records = [r for r in records if r["speaker_id"] in fresh.index]
        for condition in ["gain_0.25", "white_noise_15db"]:
            altered = perturb_records(
                fresh_records, condition, ROOT / f"cache/physiology_v2/{condition}"
            )
            table = extract_features(
                altered,
                args.extractor,
                ROOT / f"results/nisp_v2_{condition}_features.csv",
                args.reuse_features,
            )
            table["speaker_id"] = [r["speaker_id"] for r in altered]
            stressed = table.groupby("speaker_id")[names].mean().loc[fresh.index]
            for target, (model, columns) in selected_models.items():
                report["targets"][target]["stress"][condition] = {
                    "mae": mae(
                        fresh[target], model.predict(stressed[columns].to_numpy())
                    )
                }
            report["frozen_height_control"]["stress"][condition] = mae(
                fresh.height_cm, frozen_height(stressed)
            )
        for model in exported:
            result = report["targets"][model["target"]]
            clean = result["fresh_test"]
            passed = (
                result["selected"]["cv_mae"] < baseline_ratio(result)
                and clean["paired_improvement_95ci"][0] > 0
                and all(
                    s["mae"] <= clean["mae"] * 1.2 for s in result["stress"].values()
                )
            )
            result["passes_gate"] = model["passes_pilot_gate"] = bool(passed)
        save_json(ROOT / "results/nisp_expanded_report.json", report)
        save_json(
            ROOT / "results/nisp_expanded_models.json",
            {"schema_version": 2, "extractor": "sona_features_v2", "models": exported},
        )
        pd.DataFrame(predictions).to_csv(
            ROOT / "results/nisp_expanded_predictions.csv", index=False
        )
        print("Finished expanded speaker-disjoint experiment", flush=True)


def baseline_ratio(result):
    return 0.9 * result["cv_baseline_mae"]


if __name__ == "__main__":
    main()
