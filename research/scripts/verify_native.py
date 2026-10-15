#!/usr/bin/env python3
"""Check exported training parameters against real C++ inference and abstention."""

import argparse
import csv
import io
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT.parent / "build")
    args = parser.parse_args()
    model_path = ROOT / "results/nisp_pilot_models.json"
    models = json.loads(model_path.read_text())["models"]
    audio_path = ROOT / "cache/nisp_audio/Mal_0003/000.wav"
    binary = args.build_dir / "sona_traits"
    features_binary = args.build_dir / "sona_features"
    output = subprocess.check_output([str(features_binary), str(audio_path)], text=True)
    row = next(csv.DictReader(io.StringIO(output)))
    result = json.loads(
        subprocess.check_output(
            [str(binary), str(model_path), str(audio_path)], text=True
        )
    )
    maximum_error = 0.0
    for model, prediction in zip(models, result["traits"], strict=True):
        assert model["target"] == prediction["target"]
        if not model["passes_pilot_gate"]:
            assert prediction["status"] == "unsupported_by_pilot"
            assert prediction["estimate"] is None
            continue
        x = np.array([float(row[name]) for name in model["feature_names"]])
        expected = float(
            ((x - model["means"]) / model["scales"]) @ model["weights"]
            + model["intercept"]
        )
        assert prediction["status"] == "experimental_pilot_estimate"
        error = abs(prediction["estimate"] - expected)
        assert error < 1e-6, (prediction, expected)
        maximum_error = max(maximum_error, error)
    with tempfile.TemporaryDirectory() as folder:
        silence = Path(folder) / "silence.wav"
        with wave.open(str(silence), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(bytes(16000 * 3 * 2))
        quiet = json.loads(
            subprocess.check_output(
                [str(binary), str(model_path), str(silence)], text=True
            )
        )
        assert all(prediction["estimate"] is None for prediction in quiet["traits"])
        assert quiet["traits"][0]["status"] == "insufficient_clean_periodic_audio"
    report = {
        "real_wav": "NISP Mal_0003/000.wav (official test speaker)",
        "maximum_native_vs_python_absolute_difference": maximum_error,
        "unsupported_targets_abstain": True,
        "silence_abstains": True,
    }
    (ROOT / "results/native_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
