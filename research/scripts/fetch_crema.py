#!/usr/bin/env python3
"""Fetch pinned neutral utterances for a speaker-disjoint adult-age experiment."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "1658cd342dff90010aa843eaeebd53610a08b1dc"
REPOSITORY = "CheyneyComputerScience/CREMA-D"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"
MEDIA = f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/"


def read_url(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--clips-per-speaker", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.clips_per_speaker <= 12:
        parser.error("workers must be 1–32; clips-per-speaker must be 1–12")
    cache = ROOT / "cache/crema"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = read_url(RAW + "VideoDemographics.csv")
    # Retain only the age target and anonymous actor identifier.
    people = [
        {"speaker_id": row["ActorID"], "age_years": int(row["Age"])}
        for row in csv.DictReader(io.StringIO(metadata.decode()))
    ]
    (cache / "ages.json").write_text(json.dumps(people, indent=2))
    tree_path = cache / "tree.json"
    if not tree_path.exists():
        tree_path.write_bytes(
            read_url(
                f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
            )
        )
    tree = json.loads(tree_path.read_text())
    utterances = sorted(
        item["path"]
        for item in tree["tree"]
        if item["path"].startswith("AudioWAV/") and "_NEU_" in item["path"]
    )
    chosen = []
    for person in people:
        matches = [
            path
            for path in utterances
            if Path(path).name.startswith(person["speaker_id"] + "_")
        ]
        if len(matches) < args.clips_per_speaker:
            raise ValueError(
                f"Insufficient neutral utterances for {person['speaker_id']}"
            )
        chosen.extend(matches[: args.clips_per_speaker])

    def fetch(path):
        destination = cache / "audio" / Path(path).name
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            audio = read_url(MEDIA + path)
            if audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                raise ValueError(f"Not a decoded WAV: {path}")
            partial = destination.with_suffix(".partial")
            partial.write_bytes(audio)
            partial.replace(destination)
        data = destination.read_bytes()
        return {
            "speaker_id": Path(path).name.split("_")[0],
            "path": str(destination.relative_to(ROOT)),
            "source_path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(fetch, chosen))
    result = {
        "dataset": "CREMA-D",
        "revision": REVISION,
        "source": f"https://github.com/{REPOSITORY}",
        "licence": "ODbL-1.0 / DbCL-1.0",
        "condition": "performed neutral speech",
        "speaker_count": len(people),
        "recording_count": len(records),
        "people": people,
        "records": records,
    }
    output = ROOT / "results/crema_manifest.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Fetched {len(records)} neutral clips from {len(people)} speakers", flush=True
    )


if __name__ == "__main__":
    main()
