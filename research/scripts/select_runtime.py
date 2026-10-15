#!/usr/bin/env python3
"""Promote validated heads while preserving the deployed height control if needed."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import ROOT, conformal_radius, evaluate, save_json


def main():
    bank = json.loads((ROOT / "results/nisp_expanded_models.json").read_text())
    report = json.loads((ROOT / "results/nisp_expanded_report.json").read_text())
    selected = copy.deepcopy(bank)
    noise_bank = json.loads((ROOT / "results/nisp_noise_models.json").read_text())
    noise_report = json.loads((ROOT / "results/nisp_noise_report.json").read_text())
    noise_models = {m["target"]: m for m in noise_bank["models"]}
    decisions = {}
    for model in selected["models"]:
        target = model["target"]
        if target in noise_models:
            model.clear()
            model.update(copy.deepcopy(noise_models[target]))
            model["validation"]["single_utterance_interval_coverage"] = noise_report[
                "targets"
            ][target]["single_utterance"]["interval_coverage"]
        decisions[target] = (
            "enabled: noise-trained V3 head"
            if model["passes_pilot_gate"]
            else "abstain: failed predeclared gate"
        )
        # Broad-adult age has a separate, independently evaluated encoder.
        if target == "age_years":
            model["passes_pilot_gate"] = False
            decisions[target] = (
                "use separately evaluated adult-age encoder; narrow NISP age head disabled"
            )
        if target == "height_cm":
            old = copy.deepcopy(
                json.loads((ROOT / "results/nisp_pilot_models.json").read_text())[
                    "models"
                ][0]
            )
            names = model["feature_names"]
            clips = pd.read_csv(ROOT / "results/nisp_expanded_features.csv")
            clips["speaker_id"] = clips.file.map(lambda p: Path(p).parent.name)
            people = clips.groupby("speaker_id")[names].mean()
            labels = (
                pd.read_csv(ROOT / "cache/nisp_total_spkrinfo.list", sep=r"\s+")
                .set_index("Speaker_ID")
                .Height
            )
            # Old model fit every Malayalam training speaker, including some new calibration IDs.
            # Exclude that whole cohort here so no calibration observation was used in its fit.
            calibration = [
                i for i in report["calibration_ids"] if not i.startswith("Mal_")
            ]
            fresh = [identity for identity in report["fresh_test_ids"]
                     if not identity.startswith("Mal_")]

            first = clips.groupby("speaker_id").head(1).set_index("speaker_id")

            def predict(ids, frame=people):
                x = frame.loc[ids, old["feature_names"]].to_numpy()
                return ((x - old["means"]) / old["scales"]) @ np.asarray(
                    old["weights"]
                ) + old["intercept"]

            radius = conformal_radius(
                labels.loc[calibration].to_numpy(), predict(calibration, first)
            )
            old["validation"] = evaluate(
                labels.loc[fresh].to_numpy(),
                predict(fresh),
                np.full(len(fresh), labels.loc[report["fit_ids"]].median()),
                radius,
            )
            old["interval_radius"] = radius
            old["validation"]["single_utterance_mae"] = float(
                np.mean(np.abs(labels.loc[fresh].to_numpy() - predict(fresh, first)))
            )
            old["kind"] = "ridge"
            old["source_model"] = "nisp_pilot_models.json"
            old["excluded_calibration_prefixes"] = ["Mal_"]
            extension = len(names) - len(old["feature_names"])
            for key, value in [("means", 0), ("scales", 1), ("weights", 0)]:
                old[key].extend([value] * extension)
            old["feature_names"] = names
            old["trained_on"] += (
                f"; interval calibrated on {len(calibration)} new non-Malayalam speakers; validated on {len(fresh)} new non-Malayalam speakers"
            )
            model.clear()
            model.update(old)
            decisions[target] = (
                "restore original height coefficients; replacement compressed individual height differences"
            )
    save_json(ROOT / "results/nisp_runtime_models.json", selected)
    save_json(
        ROOT / "results/runtime_selection.json",
        {
            "decisions": decisions,
            "rule": "Height uses the original frozen coefficients after a reported live regression. The noise-trained height remains an explicit comparison in nisp_noise_models.json. Other heads retain their registered validation gates.",
        },
    )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
