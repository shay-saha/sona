#!/usr/bin/env python3
"""Audit height confounding using provided metadata, never an inferred demographic."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = pd.read_csv(ROOT / "cache/nisp_total_spkrinfo.list", sep=r"\s+")
    metadata = metadata.set_index("Speaker_ID")
    train_ids = {
        value.split("___")[0]
        for value in (ROOT / "cache/nisp_train_spkrID").read_text().split()
    }
    available = pd.read_csv(ROOT / "results/nisp_pilot_features.csv")
    available_ids = {Path(value).parent.name for value in available.file}
    train = metadata.loc[metadata.index.isin(train_ids & available_ids)]
    predictions = pd.read_csv(ROOT / "results/nisp_pilot_predictions.csv")
    predictions = predictions[predictions.target == "height_cm"].set_index("speaker_id")
    test = metadata.loc[predictions.index].copy()
    medians = train.groupby("Gender").Height.median()
    oracle = test.Gender.map(medians).to_numpy()
    truth = predictions.truth.to_numpy()
    audio = predictions.prediction.to_numpy()
    result = {
        "audit_only": "Provided sex metadata is used only for aggregate confound analysis; never as a model input or prediction target.",
        "oracle_group_median_mae_cm": float(np.mean(np.abs(truth - oracle))),
        "audio_model_mae_cm": float(np.mean(np.abs(truth - audio))),
        "groups": {},
    }
    for group in sorted(test.Gender.unique()):
        mask = test.Gender.to_numpy() == group
        result["groups"][group] = {
            "test_speakers": int(mask.sum()),
            "audio_model_mae_cm": float(np.mean(np.abs(truth[mask] - audio[mask]))),
            "provided_group_constant_mae_cm": float(
                np.mean(np.abs(truth[mask] - oracle[mask]))
            ),
        }
    (ROOT / "results/nisp_confound_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
