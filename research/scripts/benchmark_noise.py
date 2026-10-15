#!/usr/bin/env python3
"""Noise-trained regressors, with V2 results and official test speakers frozen."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from benchmark_physiology import (
    COHORTS,
    TARGETS,
    candidates,
    export_model,
    groups,
    pipeline,
)
from experiment_utils import (
    ROOT,
    conformal_radius,
    evaluate,
    extract_features,
    mae,
    paired_interval,
    perturb_records,
    save_json,
)

SEED = 20261017
TRAIN_CONDITIONS = ["clean", "white_noise_10db", "white_noise_20db"]


def condition_features(records, condition, extractor, reuse):
    altered = (
        records
        if condition == "clean"
        else perturb_records(
            records, condition, ROOT / f"cache/physiology_v3/{condition}"
        )
    )
    table = extract_features(
        altered,
        extractor,
        ROOT / f"results/nisp_v3_{condition}_features.csv",
        reuse=reuse,
    )
    table["speaker_id"] = [r["speaker_id"] for r in altered]
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--extractor", type=Path, default=ROOT.parent / "build/sona_features"
    )
    args = parser.parse_args()
    previous = json.loads((ROOT / "results/nisp_expanded_report.json").read_text())
    protocol = json.loads((ROOT / "experiments/physiology_noise_v3.json").read_text())
    fit_ids, cal_ids = train_test_split(
        previous["fit_ids"],
        test_size=0.2,
        random_state=SEED,
        stratify=[i.split("_")[0] for i in previous["fit_ids"]],
    )
    test_ids = previous["calibration_ids"]
    assert len(fit_ids) == 134 and len(cal_ids) == 34 and len(test_ids) == 42
    assert not set(fit_ids) & set(cal_ids) and not (set(fit_ids) | set(cal_ids)) & set(
        test_ids
    )
    records = []
    for cohort in COHORTS:
        records.extend(
            json.loads(
                (ROOT / f"results/nisp_manifest_{cohort}_master.json").read_text()
            )["records"]
        )
    training_records = [
        r for r in records if r["speaker_id"] in set(fit_ids) | set(cal_ids)
    ]
    test_records = [r for r in records if r["speaker_id"] in test_ids]
    clean = pd.read_csv(ROOT / "results/nisp_expanded_features.csv")
    names = clean.columns[1:].tolist()
    clean["speaker_id"] = clean.file.map(lambda p: Path(p).parent.name)
    tables = {"clean": clean}
    for condition in TRAIN_CONDITIONS[1:]:
        tables[condition] = condition_features(
            training_records, condition, args.extractor, args.reuse_features
        )
    people = {
        condition: table.groupby("speaker_id")[names].mean()
        for condition, table in tables.items()
    }
    labels = pd.read_csv(ROOT / "cache/nisp_total_spkrinfo.list", sep=r"\s+").set_index(
        "Speaker_ID"
    )
    fit = pd.concat([people[c].loc[fit_ids] for c in TRAIN_CONDITIONS])
    fit_speakers = fit.index.to_numpy()
    folds = list(GroupKFold(5).split(fit, groups=[i.split("_")[0] for i in fit.index]))
    for train, val in folds:
        assert not set(fit_speakers[train]) & set(fit_speakers[val])
    report = {
        "protocol": protocol,
        "fit_ids": fit_ids,
        "calibration_ids": cal_ids,
        "test_ids": test_ids,
        "targets": {},
        "perturbation_seed": 20261016,
    }
    selected, exported, predictions = {}, [], []
    with threadpool_limits(limits=1):
        for target in protocol["targets"]:
            column, unit = TARGETS[target]
            y = labels.loc[fit.index, column].to_numpy(dtype=float)
            yt = labels.loc[test_ids, column].to_numpy(dtype=float)
            yc = labels.loc[cal_ids, column].to_numpy(dtype=float)
            baseline_oof = np.zeros(len(y))
            for train, val in folds:
                baseline_oof[val] = np.median(y[train])
            rows = []
            for group in protocol["groups"]:
                columns = groups(names)[group]
                x = fit[columns].to_numpy()
                for config in candidates():
                    oof = np.zeros(len(y))
                    for train, val in folds:
                        model = pipeline(config, len(columns)).fit(x[train], y[train])
                        oof[val] = model.predict(x[val])
                    rows.append(
                        {
                            "group": group,
                            "columns": columns,
                            "config": config,
                            "cv_mae": mae(y, oof),
                        }
                    )
            best = min(rows, key=lambda row: row["cv_mae"])
            columns = best["columns"]
            model = pipeline(best["config"], len(columns)).fit(
                fit[columns].to_numpy(), y
            )
            selected[target] = (model, columns)
            calibration_errors = []
            for condition in TRAIN_CONDITIONS:
                first = (
                    tables[condition]
                    .groupby("speaker_id")
                    .head(1)
                    .set_index("speaker_id")
                    .loc[cal_ids]
                )
                calibration_errors.append(
                    np.abs(yc - model.predict(first[columns].to_numpy()))
                )
            maximum_errors = np.max(calibration_errors, axis=0)
            radius = conformal_radius(maximum_errors, np.zeros(len(yc)))
            pred = model.predict(people["clean"].loc[test_ids, columns].to_numpy())
            baseline = float(np.median(y))
            first_test = (
                clean.groupby("speaker_id")
                .head(1)
                .set_index("speaker_id")
                .loc[test_ids]
            )
            first_pred = model.predict(first_test[columns].to_numpy())
            result = {
                "selected": best,
                "candidates": rows,
                "cv_baseline_mae": mae(y, baseline_oof),
                "clean": evaluate(yt, pred, np.full(len(yt), baseline), radius),
                "stress": {},
                "single_utterance": evaluate(
                    yt, first_pred, np.full(len(yt), baseline), radius
                ),
            }
            result["clean"]["single_utterance_mae"] = result["single_utterance"]["mae"]
            result["clean"]["single_utterance_interval_coverage"] = result[
                "single_utterance"
            ]["interval_coverage"]
            report["targets"][target] = result
            exported.append(
                {
                    **export_model(model, columns, names),
                    "target": target,
                    "unit": unit,
                    "passes_pilot_gate": False,
                    "interval_radius": radius,
                    "validation": result["clean"],
                    "trained_on": "NISP English; 134 fit / 34 calibration / 42 test speakers; simulated 10/20 dB training noise",
                    "protocol": "physiology_noise_v3; test speakers previously calibrated V2 intervals only",
                }
            )
            for speaker, truth, prediction in zip(test_ids, yt, pred):
                predictions.append(
                    {
                        "speaker_id": speaker,
                        "target": target,
                        "truth": float(truth),
                        "prediction": float(prediction),
                    }
                )
            print(target, best["group"], best["config"], result["clean"], flush=True)
        for condition in ["gain_0.25", "white_noise_15db"]:
            table = condition_features(
                test_records, condition, args.extractor, args.reuse_features
            )
            stressed = table.groupby("speaker_id")[names].mean().loc[test_ids]
            first = (
                table.groupby("speaker_id")
                .head(1)
                .set_index("speaker_id")
                .loc[test_ids]
            )
            for target, (model, columns) in selected.items():
                truth = labels.loc[test_ids, TARGETS[target][0]].to_numpy(dtype=float)
                pred = model.predict(stressed[columns].to_numpy())
                radius = next(
                    m["interval_radius"] for m in exported if m["target"] == target
                )
                report["targets"][target]["stress"][condition] = {
                    "mae": mae(truth, pred),
                    "single_utterance_mae": mae(
                        truth, model.predict(first[columns].to_numpy())
                    ),
                    "interval_coverage": float(np.mean(np.abs(truth - pred) <= radius)),
                }
        for model in exported:
            result = report["targets"][model["target"]]
            result["passes_gate"] = model["passes_pilot_gate"] = bool(
                result["selected"]["cv_mae"] < result["cv_baseline_mae"] * 0.9
                and result["clean"]["paired_improvement_95ci"][0] > 0
                and all(
                    s["mae"] <= result["clean"]["mae"] * 1.2
                    for s in result["stress"].values()
                )
            )
        old = json.loads((ROOT / "results/nisp_pilot_models.json").read_text())[
            "models"
        ][0]
        ids = [i for i in test_ids if not i.startswith("Mal_")]
        old_x = people["clean"].loc[ids, old["feature_names"]].to_numpy()
        old_pred = ((old_x - old["means"]) / old["scales"]) @ np.asarray(
            old["weights"]
        ) + old["intercept"]
        new_model, columns = selected["height_cm"]
        new_pred = new_model.predict(people["clean"].loc[ids, columns].to_numpy())
        report["frozen_height_control"] = {
            "speakers": len(ids),
            "mae": mae(labels.loc[ids, "Height"], old_pred),
            "new_model_improvement_95ci": paired_interval(
                labels.loc[ids, "Height"].to_numpy(), old_pred, new_pred
            ),
        }
    save_json(
        ROOT / "results/nisp_noise_models.json",
        {"schema_version": 2, "extractor": "sona_features_v2", "models": exported},
    )
    save_json(ROOT / "results/nisp_noise_report.json", report)
    pd.DataFrame(predictions).to_csv(
        ROOT / "results/nisp_noise_predictions.csv", index=False
    )
    print(
        json.dumps(
            {
                t: {
                    "clean": r["clean"]["mae"],
                    "stress": r["stress"],
                    "passes_gate": r["passes_gate"],
                }
                for t, r in report["targets"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
