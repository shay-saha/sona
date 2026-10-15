"""Shared speaker-level evaluation and native feature extraction."""

import concurrent.futures
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261016


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def extract_features(records, extractor, destination, reuse=False):
    paths = [record["path"] for record in records]
    cached = None
    if reuse and destination.exists():
        cached = pd.read_csv(destination).set_index("file", drop=False)
        if not cached.index.is_unique:
            raise ValueError("Duplicate cached audio paths")
        if cached.index.tolist() == paths:
            return cached.reset_index(drop=True)
    pending = [p for p in paths if cached is None or p not in cached.index]

    def run(chunk):
        result = subprocess.run(
            [str(extractor.resolve()), "--extended", *chunk],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return pd.read_csv(io.StringIO(result.stdout))

    chunks = [pending[i : i + 48] for i in range(0, len(pending), 48)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = [] if cached is None else [cached.reset_index(drop=True)]
        for part in pool.map(run, chunks):
            parts.append(part)
            print(f"Descriptors: {sum(len(p) for p in parts)}/{len(paths)}", flush=True)
    table = (
        pd.concat(parts, ignore_index=True)
        .set_index("file", drop=False)
        .loc[paths]
        .reset_index(drop=True)
    )
    if table.file.tolist() != paths or not np.isfinite(table.iloc[:, 1:].values).all():
        raise ValueError("Invalid native feature output")
    table.to_csv(destination, index=False)
    return table


def mae(y, predictions):
    return float(mean_absolute_error(y, predictions))


def paired_interval(y, baseline, predictions):
    differences = np.abs(y - baseline) - np.abs(y - predictions)
    rng = np.random.default_rng(SEED)
    boot = rng.choice(differences, (5000, len(y)), replace=True).mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def conformal_radius(y, predictions):
    residuals = np.sort(np.abs(y - predictions))
    rank = min(len(residuals), math.ceil((len(residuals) + 1) * 0.9))
    return float(residuals[rank - 1])


def evaluate(y, predictions, baseline, radius):
    return {
        "speakers": len(y),
        "mae": mae(y, predictions),
        "baseline_mae": mae(y, baseline),
        "paired_improvement_95ci": paired_interval(y, baseline, predictions),
        "nominal_coverage": 0.9,
        "interval_radius": radius,
        "interval_coverage": float(np.mean(np.abs(y - predictions) <= radius)),
    }


def perturb_records(records, condition, folder):
    """Fixed full-recording-RMS SNR, without a hidden denoising/level change."""
    import hashlib
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    outputs = []
    for record in records:
        source = ROOT / record["path"]
        key = hashlib.sha256(record["path"].encode()).hexdigest()[:20]
        destination = folder / f"{key}.wav"
        if not destination.exists():
            samples, rate = sf.read(source, dtype="float32")
            if condition == "gain_0.25":
                samples = samples * 0.25
            elif condition.startswith("white_noise_") and condition.endswith("db"):
                rng = np.random.default_rng(int(key[:8], 16) + SEED)
                noise = rng.normal(size=samples.shape)
                snr = float(condition.removeprefix("white_noise_").removesuffix("db"))
                amplitude = np.sqrt(np.mean(samples.astype(float) ** 2)) / 10 ** (
                    snr / 20
                )
                samples = samples + noise * amplitude
            else:
                raise ValueError(condition)
            sf.write(destination, samples, rate, subtype="FLOAT")
        outputs.append({**record, "path": str(destination.relative_to(ROOT))})
    return outputs
