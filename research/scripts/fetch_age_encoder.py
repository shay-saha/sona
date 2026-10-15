#!/usr/bin/env python3
"""Prepare the published six-layer speech encoder with its age output only.

The released network has multiple heads. Prune all outputs and graph branches
except age before inference. We never run or expose a gender classifier.
The source weights are CC BY-NC-SA 4.0 and remain in the ignored cache.
"""

import hashlib
import json
import concurrent.futures
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip"
URL = f"https://zenodo.org/api/records/7761387/files/{NAME}/content"
EXPECTED_MD5 = "d1479956f0cf50107198276dc1b4f1d0"
ARCHIVE_BYTES = 332874239


def download_archive(destination):
    partial = destination.with_suffix(".partial")
    prefix = partial.stat().st_size if partial.exists() else 0
    if prefix > ARCHIVE_BYTES:
        raise ValueError("Oversize partial download")
    chunks = destination.parent / "parts"
    chunks.mkdir(exist_ok=True)
    spans = [
        (start, min(start + 8 * 1024 * 1024, ARCHIVE_BYTES) - 1)
        for start in range(prefix, ARCHIVE_BYTES, 8 * 1024 * 1024)
    ]

    def fetch(span):
        start, end = span
        path = chunks / f"{start}-{end}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    URL + f"?segment={start}", headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{ARCHIVE_BYTES}"
                    ):
                        raise ValueError("Server did not honor exact byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Truncated download range")
                path.write_bytes(data)
                print(f"Downloaded bytes {start}–{end}", flush=True)
                return path
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(fetch, spans))
    with partial.open("ab") as stream:
        for path in parts:
            stream.write(path.read_bytes())
    partial.replace(destination)


def main():
    import onnx

    folder = ROOT / "cache/age_encoder"
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = folder / NAME
    if not archive_path.exists():
        download_archive(archive_path)
    digest = hashlib.md5()
    with archive_path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_MD5:
        raise ValueError("Age encoder archive checksum mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        print("Archive members:", archive.namelist(), flush=True)
        candidates = [name for name in archive.namelist() if name.endswith(".onnx")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one ONNX graph")
        source_path = folder / "source.onnx"
        with archive.open(candidates[0]) as source, source_path.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
    graph = onnx.load(source_path)
    inputs = [value.name for value in graph.graph.input]
    outputs = [value.name for value in graph.graph.output]
    print("Graph inputs:", inputs, "outputs:", outputs, flush=True)
    age_outputs = [name for name in outputs if "age" in name.lower()]
    if len(age_outputs) != 1:
        raise ValueError("Ambiguous age output; inspect source model metadata")
    destination = folder / "age_only.onnx"
    onnx.utils.extract_model(str(source_path), str(destination), inputs, age_outputs)
    pruned = onnx.load(destination)
    assert [value.name for value in pruned.graph.output] == age_outputs
    provenance = {
        "source": "https://zenodo.org/records/7761387",
        "paper": "https://arxiv.org/abs/2306.16962",
        "authors": [
            "Felix Burkhardt",
            "Johannes Wagner",
            "Hagen Wierstorf",
            "Florian Eyben",
            "Björn Schuller",
        ],
        "archive_md5": EXPECTED_MD5,
        "licence": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "modification": "Age output retained; other output branches removed before inference",
        "inputs": inputs,
        "outputs": age_outputs,
        "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Prepared", destination, flush=True)


if __name__ == "__main__":
    main()
