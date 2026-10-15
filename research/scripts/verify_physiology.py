#!/usr/bin/env python3
"""Compare exported Python regressors and native inference on real waveforms."""

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from age_inference import AgeEncoder
from experiment_utils import ROOT, save_json


def run(arguments):
    return json.loads(subprocess.check_output([str(a) for a in arguments], text=True))


def regression(model, features):
    x = (features - np.asarray(model["means"])) / np.asarray(model["scales"])
    if model["kind"] == "ridge":
        return float(x @ np.asarray(model["weights"]) + model["intercept"])
    squared = np.sum(
        (np.asarray(model["support_vectors"]) - x[model["feature_indices"]]) ** 2,
        axis=1,
    )
    return float(
        np.exp(-model["gamma"] * squared) @ np.asarray(model["dual_coefficients"])
        + model["intercept"]
    )


def check_regressors(build, temporary, silence, filename):
    manifest = json.loads(
        (ROOT / "results/nisp_manifest_Malayalam_master.json").read_text()
    )
    features = pd.read_csv(ROOT / "results/nisp_expanded_features.csv").set_index(
        "file"
    )
    bank = json.loads((ROOT / "results" / filename).read_text())
    # Test-only gates permit arithmetic checks even for rejected research candidates.
    fixture = copy.deepcopy(bank)
    for model in fixture["models"]:
        model["passes_pilot_gate"] = True
        model["trained_on"] = "Numerical parity fixture only; not a deployment model"
    errors = []
    model_path = temporary / "regressors.json"
    save_json(model_path, fixture)
    for record in manifest["records"][:18]:
        actual = run([build / "sona_traits", model_path, ROOT / record["path"]])
        x = features.loc[record["path"]].to_numpy(dtype=float)
        for p, model in zip(actual["traits"], bank["models"]):
            if p["estimate"] is not None:
                expected = regression(model, x)
                errors.append(abs(expected - p["estimate"]))
                assert abs(expected - p["estimate"]) < 1e-6
                assert (
                    abs(
                        p["interval_90_upper"]
                        - p["estimate"]
                        - model["interval_radius"]
                    )
                    < 1e-6
                )
    if len(errors) < 4:
        raise ValueError("Not enough eligible physiology parity comparisons")
    assert all(
        p["estimate"] is None
        for p in run([build / "sona_traits", model_path, silence])["traits"]
    )
    return errors


def check_age(build, temporary, silence):
    if not run([build / "sona_age", "--capabilities"])["age_runtime"]:
        return {"age_runtime": "not built", "age_comparisons": 0}
    calibration = json.loads((ROOT / "results/age_calibration.json").read_text())
    calibration["passes_gate"] = True
    calibration["scope"] = "Numerical parity fixture only; not a deployment model"
    cal_path = temporary / "age_fixture.json"
    save_json(cal_path, calibration)
    graph = ROOT / "cache/age_encoder/age_only.onnx"
    encoder = AgeEncoder(graph)
    crema = pd.read_csv(ROOT / "results/crema_features.csv")
    eligible = crema.loc[
        (crema.duration_s >= 2)
        & (crema.active_fraction >= 0.3)
        & (crema.periodic_fraction >= 0.2)
        & (crema.clipped_fraction <= 0.01)
    ]
    age_errors = []
    for path in eligible.file.head(4):
        expected = (
            encoder.predict(ROOT / path) * calibration["slope"]
            + calibration["intercept"]
        )
        actual = run([build / "sona_age", graph, cal_path, ROOT / path])
        if actual["estimate"] is not None:
            age_errors.append(abs(actual["estimate"] - expected))
    if not age_errors or max(age_errors) > 0.02:
        raise ValueError(f"Age runtime disagreement: {age_errors}")
    assert run([build / "sona_age", graph, cal_path, silence])["estimate"] is None
    if not json.loads((ROOT / "results/age_calibration.json").read_text())[
        "passes_gate"
    ]:
        rejected = run(
            [
                build / "sona_age",
                graph,
                ROOT / "results/age_calibration.json",
                ROOT / eligible.file.iloc[0],
            ]
        )
        assert (
            rejected["estimate"] is None
            and rejected["status"] == "unsupported_by_pilot"
        )
    calibration["model_sha256"] = "0" * 64
    save_json(cal_path, calibration)
    bad_hash = subprocess.run(
        [str(build / "sona_age"), str(graph), str(cal_path), str(silence)],
        capture_output=True,
    )
    assert bad_hash.returncode != 0 and b"checksum mismatch" in bad_hash.stderr
    return {
        "age_runtime": "verified",
        "age_comparisons": len(age_errors),
        "maximum_age_error_years": max(age_errors),
        "failed_age_gate_abstention": True,
        "encoder_checksum_guard": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT.parent / "build")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    with tempfile.TemporaryDirectory(
        dir=ROOT / "cache", prefix="native-parity-"
    ) as directory:
        temporary = Path(directory)
        silence = temporary / "silence.wav"
        sf.write(silence, np.zeros(48000, dtype=np.float32), 16000, subtype="FLOAT")
        errors = []
        for filename in [
            "nisp_expanded_models.json",
            "nisp_noise_models.json",
            "nisp_runtime_models.json",
        ]:
            errors.extend(check_regressors(build, temporary, silence, filename))
        report = {
            "physiology_comparisons": len(errors),
            "maximum_regression_error": max(errors),
            "silence_abstention": True,
            **check_age(build, temporary, silence),
        }
    save_json(ROOT / "results/physiology_native_verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
