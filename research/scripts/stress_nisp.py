#!/usr/bin/env python3
"""Unchanged pilot models under controlled gain/noise perturbations; no retraining."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractor", type=Path, default=ROOT.parent / "build/sona_features"
    )
    args = parser.parse_args()
    report = json.loads((ROOT / "results/nisp_pilot_report.json").read_text())
    model_data = json.loads((ROOT / "results/nisp_pilot_models.json").read_text())
    clean = pd.read_csv(ROOT / "results/nisp_pilot_features.csv")
    clean["speaker_id"] = clean.file.map(lambda p: Path(p).parent.name)
    clean = clean[clean.speaker_id.isin(report["test_speaker_ids"])]
    folder = ROOT / "cache/stress"
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths = []
    for record in clean.itertuples():
        audio, rate = sf.read(ROOT / record.file, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        power = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
        noise = rng.standard_normal(len(audio))
        noise *= power / (10 ** (15 / 20) * np.sqrt(np.mean(noise**2)))
        for condition, signal in [
            ("gain_minus_12db", audio * 0.25),
            ("white_noise_15db_snr", audio + noise),
        ]:
            path = folder / condition / record.speaker_id / Path(record.file).name
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, signal, rate, subtype="FLOAT")
            paths.append(str(path.relative_to(ROOT)))
    feature_path = ROOT / "results/nisp_stress_features.csv"
    print(f"Extracting {len(paths)} perturbed recordings", flush=True)
    with feature_path.open("w") as output:
        subprocess.run(
            [str(args.extractor.resolve()), *paths], cwd=ROOT, stdout=output, check=True
        )
    stress = pd.read_csv(feature_path)
    stress["speaker_id"] = stress.file.map(lambda p: Path(p).parent.name)
    stress["condition"] = stress.file.map(lambda p: Path(p).parent.parent.name)
    predictions = pd.read_csv(ROOT / "results/nisp_pilot_predictions.csv")
    result = {
        "conditions": {
            "gain_minus_12db": "Same waveform multiplied by 0.25; approximately -12.04 dB.",
            "white_noise_15db_snr": "Generated white Gaussian noise at 15 dB full-recording RMS SNR; laboratory stress test, not real-room validation.",
        },
        "targets": {},
    }
    names = model_data["models"][0]["feature_names"]
    people = clean.groupby("speaker_id")[names].mean()
    for model in model_data["models"]:
        target = model["target"]
        index = report["test_speaker_ids"]
        truth = (
            predictions[predictions.target == target]
            .set_index("speaker_id")
            .loc[index, "truth"]
            .to_numpy()
        )
        weights = np.asarray(model["weights"])
        means = np.asarray(model["means"])
        scales = np.asarray(model["scales"])

        def predict(frame):
            return ((frame[names].to_numpy() - means) / scales) @ weights + model[
                "intercept"
            ]

        original = predict(people.loc[index])
        results = {"clean_mae": float(mean_absolute_error(truth, original))}
        for condition in result["conditions"]:
            observed = (
                stress[stress.condition == condition]
                .groupby("speaker_id")[names]
                .mean()
                .loc[index]
            )
            p = predict(observed)
            results[condition] = {
                "mae": float(mean_absolute_error(truth, p)),
                "median_absolute_prediction_shift": float(
                    np.median(np.abs(p - original))
                ),
                "max_absolute_prediction_shift": float(np.max(np.abs(p - original))),
            }
        result["targets"][target] = results
        print(target, results, flush=True)
    (ROOT / "results/nisp_stress_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
