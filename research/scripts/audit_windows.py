#!/usr/bin/env python3
"""Audit frozen regressors on the short windows used by the native application."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from benchmark_physiology import COHORTS, TARGETS
from experiment_utils import (
    ROOT,
    conformal_radius,
    extract_features,
    mae,
    paired_interval,
    save_json,
)
from verify_physiology import regression


def windows(records, ids, condition, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = []
    for speaker in ids:
        selected = [r for r in records if r["speaker_id"] == speaker]
        parts = []
        for record in selected:
            signal, rate = sf.read(ROOT / record["path"], dtype="float32")
            if signal.ndim == 2:
                signal = signal.mean(axis=1)
            factor = gcd(rate, 16000)
            if rate != 16000:
                signal = resample_poly(signal, 16000 // factor, rate // factor).astype(
                    np.float32
                )
            parts.append(signal)
        reference = np.concatenate(parts)[:160000]
        if len(reference) < 160000:
            raise ValueError(f"Need ten seconds of held-out speech: {speaker}")
        for seconds in [2, 5, 10]:
            signal = reference[-seconds * 16000 :].copy()
            if condition == "gain_0.25":
                signal *= 0.25
            elif condition.startswith("white_noise_"):
                snr = float(condition.removeprefix("white_noise_").removesuffix("db"))
                seed = int(
                    hashlib.sha256(f"{speaker}/{seconds}".encode()).hexdigest()[:8], 16
                )
                rng = np.random.default_rng(seed + 20261017)
                amplitude = np.sqrt(np.mean(signal.astype(float) ** 2)) / 10 ** (
                    snr / 20
                )
                signal = signal + rng.normal(size=signal.shape) * amplitude
            elif condition != "clean":
                raise ValueError(condition)
            path = folder / f"{speaker}_{seconds}s.wav"
            if not path.exists():
                sf.write(path, signal, 16000, subtype="FLOAT")
            result.append(
                {
                    "speaker_id": speaker,
                    "seconds": seconds,
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return result


def eligible_windows(model, frame, names):
    eligible = (
        (frame.duration_s >= 2)
        & (frame.active_fraction >= 0.3)
        & (frame.periodic_fraction >= 0.2)
        & (frame.clipped_fraction <= 0.01)
    ).to_numpy()
    used = (
        np.flatnonzero(model["weights"])
        if model["kind"] == "ridge"
        else model["feature_indices"]
    )
    z = (frame[names].to_numpy() - model["means"]) / model["scales"]
    return eligible & (np.abs(z[:, used]) <= 8).all(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extractor", type=Path, default=ROOT.parent / "build/sona_features"
    )
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    frozen = json.loads((ROOT / "results/nisp_noise_report.json").read_text())
    bank = json.loads((ROOT / "results/nisp_runtime_models.json").read_text())
    records = []
    for cohort in COHORTS:
        records.extend(
            json.loads(
                (ROOT / f"results/nisp_manifest_{cohort}_master.json").read_text()
            )["records"]
        )
    names = bank["models"][0]["feature_names"]
    tables = {}
    for split, ids, conditions in [
        (
            "calibration",
            frozen["calibration_ids"],
            ["clean", "white_noise_10db", "white_noise_20db"],
        ),
        ("test", frozen["test_ids"], ["clean", "gain_0.25", "white_noise_15db"]),
    ]:
        for condition in conditions:
            selected = windows(
                records, ids, condition, ROOT / f"cache/windows/{split}/{condition}"
            )
            table = extract_features(
                selected,
                args.extractor,
                ROOT / f"results/windows_{split}_{condition}_features.csv",
                args.reuse_features,
            )
            table["speaker_id"] = [r["speaker_id"] for r in selected]
            table["seconds"] = [r["seconds"] for r in selected]
            tables[(split, condition)] = table
    labels = pd.read_csv(ROOT / "cache/nisp_total_spkrinfo.list", sep=r"\s+").set_index(
        "Speaker_ID"
    )
    report = {
        "protocol": "Frozen point-model audit; no coefficient, feature or hyperparameter changes. Concatenate the six selected utterances, take the first ten seconds, then nested trailing 2/5/10-second windows. Each calibration identity contributes its maximum residual across windows passing the existing native quality and feature-range gates; identities with no eligible windows are excluded. Separate calibration speakers set interval radii before scoring test windows.",
        "calibration_ids": frozen["calibration_ids"],
        "test_ids": frozen["test_ids"],
        "targets": {},
    }
    for model in bank["models"]:
        if not model["passes_pilot_gate"]:
            continue
        target, column = model["target"], TARGETS[model["target"]][0]
        calibration_ids = [identity for identity in frozen["calibration_ids"]
                           if not any(identity.startswith(prefix) for prefix in
                                      model.get("excluded_calibration_prefixes", []))]
        evaluation_ids = [identity for identity in frozen["test_ids"]
                          if not any(identity.startswith(prefix) for prefix in
                                     model.get("excluded_calibration_prefixes", []))]
        residuals = []
        for condition in ["clean", "white_noise_10db", "white_noise_20db"]:
            table = tables[("calibration", condition)]
            for seconds in [2, 5, 10]:
                subset = (
                    table.loc[table.seconds == seconds]
                    .set_index("speaker_id")
                    .loc[calibration_ids]
                )
                p = np.array(
                    [regression(model, row) for row in subset[names].to_numpy()]
                )
                residuals.append(
                    np.abs(labels.loc[subset.index, column].to_numpy() - p)
                )
                residuals[-1][~eligible_windows(model, subset, names)] = np.nan
        maximum = np.array(
            [
                np.max(values[np.isfinite(values)])
                if np.isfinite(values).any()
                else np.nan
                for values in np.asarray(residuals).T
            ]
        )
        maximum = maximum[np.isfinite(maximum)]
        radius = conformal_radius(maximum, np.zeros(len(maximum)))
        result = {
            "interval_radius": radius,
            "eligible_calibration_speakers": len(maximum),
            "interval_calibration_ids": calibration_ids,
            "evaluation_ids": evaluation_ids,
            "conditions": {},
        }
        baseline_value = float(labels.loc[frozen["fit_ids"], column].median())
        for condition in ["clean", "gain_0.25", "white_noise_15db"]:
            result["conditions"][condition] = {}
            table = tables[("test", condition)]
            for seconds in [2, 5, 10]:
                subset = (
                    table.loc[table.seconds == seconds]
                    .set_index("speaker_id")
                    .loc[evaluation_ids]
                )
                truth = labels.loc[subset.index, column].to_numpy()
                p = np.array(
                    [regression(model, row) for row in subset[names].to_numpy()]
                )
                eligible = eligible_windows(model, subset, names)
                result["conditions"][condition][str(seconds)] = {
                    "speakers": len(truth),
                    "mae": mae(truth, p),
                    "baseline_mae": mae(truth, np.full(len(truth), baseline_value)),
                    "paired_improvement_95ci": paired_interval(
                        truth, np.full(len(truth), baseline_value), p
                    ),
                    "interval_coverage": float(np.mean(np.abs(truth - p) <= radius)),
                    "eligible_speakers": int(eligible.sum()),
                    "eligible_mae": mae(truth[eligible], p[eligible])
                    if eligible.any()
                    else None,
                    "eligible_interval_coverage": float(
                        np.mean(np.abs(truth[eligible] - p[eligible]) <= radius)
                    )
                    if eligible.any()
                    else None,
                }
        report["targets"][target] = result
        # Only the uncertainty metadata changes; fitted point-model parameters remain frozen.
        model["interval_radius"] = radius
        live = result["conditions"]["clean"]["10"]
        model["validation"]["single_utterance_mae"] = live["eligible_mae"]
        model["validation"]["single_utterance_interval_coverage"] = live[
            "eligible_interval_coverage"
        ]
        marker = "; interval audited on eligible 2/5/10-second windows"
        model["trained_on"] = model["trained_on"].replace(
            "; interval audited on 2/5/10-second windows", ""
        )
        if marker not in model["trained_on"]:
            model["trained_on"] += marker
    save_json(ROOT / "results/live_window_report.json", report)
    save_json(ROOT / "results/nisp_runtime_models.json", bank)
    print(
        json.dumps(
            {t: r["conditions"] for t, r in report["targets"].items()}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
