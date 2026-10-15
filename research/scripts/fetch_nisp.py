#!/usr/bin/env python3
"""Fetch the authors' pinned NISP archives; keep only a bounded WAV pilot subset."""

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "d8ee6a15e1c188ca649b0c5d80a4dcde673882fe"
BASE = f"https://raw.githubusercontent.com/iiscleap/NISP-Dataset/{REVISION}/"


def blob_digest(path):
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_part(item, folder):
    destination = folder / Path(item["path"]).name
    if (
        destination.exists()
        and destination.stat().st_size == item["size"]
        and blob_digest(destination) == item["sha"]
    ):
        return destination
    destination.unlink(missing_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha1(f"blob {item['size']}\0".encode())
    total = partial.stat().st_size if partial.exists() else 0
    if total:
        with partial.open("rb") as previous:
            while block := previous.read(4 * 1024 * 1024):
                digest.update(block)
    headers = {"User-Agent": "SONA-research/0.1"}
    if total:
        headers["Range"] = f"bytes={total}-"
    request = urllib.request.Request(BASE + item["path"], headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if total and (
            response.status != 206
            or not response.headers.get("Content-Range", "").startswith(
                f"bytes {total}-"
            )
        ):
            raise RuntimeError("Server did not honor the resume range")
        with partial.open("ab" if total else "wb") as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
                total += len(block)
    if total != item["size"] or digest.hexdigest() != item["sha"]:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch: {item['path']}")
    partial.replace(destination)
    print(f"Verified {destination.name}: {total / 1e6:.1f} MB", flush=True)
    return destination


class JoinedFiles(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        view = memoryview(buffer)
        count = 0
        while count < len(view):
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            n = self.current.readinto(view[count:])
            if n:
                count += n
            else:
                self.current.close()
                self.current = None
        return count

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="Malayalam_master/English_Malayalam")
    parser.add_argument("--clips-per-speaker", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.clips_per_speaker <= 100:
        parser.error("clips-per-speaker must be 1–100")
    if not 1 <= args.workers <= 32:
        parser.error("workers must be 1–32")
    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    tree_path = cache / "nisp_tree.json"
    if not tree_path.exists():
        data = urllib.request.urlopen(
            f"https://api.github.com/repos/iiscleap/NISP-Dataset/git/trees/{REVISION}?recursive=1",
            timeout=30,
        ).read()
        tree_path.write_bytes(data)
    tree = json.loads(tree_path.read_text())
    if tree["sha"] != REVISION:
        raise RuntimeError("Unexpected repository revision")
    items = sorted(
        (x for x in tree["tree"] if x["path"].startswith(args.group + "/RECS.tar.gz.")),
        key=lambda x: x["path"],
    )
    if not items:
        parser.error("Unknown archive group")
    folder = cache / args.group.replace("/", "_")
    folder.mkdir(exist_ok=True)
    print(
        f"Pinned NISP {REVISION}; {len(items)} parts, {sum(x['size'] for x in items) / 1e9:.2f} GB",
        flush=True,
    )
    for name in ["total_spkrinfo.list", "train_spkrID", "test_spkrID", "README.md"]:
        (cache / ("nisp_" + name)).write_bytes(
            urllib.request.urlopen(BASE + name, timeout=30).read()
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        paths = list(executor.map(lambda x: fetch_part(x, folder), items))
    counts = {}
    records = []
    target = cache / "nisp_audio"
    target.mkdir(exist_ok=True)
    # Read archive members as streams. Never unpack archive paths onto the filesystem.
    with io.BufferedReader(JoinedFiles(paths), buffer_size=1024 * 1024) as joined:
        # The release is gzip-wrapped twice, as reflected by the authors' extraction command.
        with gzip.GzipFile(fileobj=joined, mode="rb") as decoded:
            with tarfile.open(fileobj=decoded, mode="r|*") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.lower().endswith(".wav"):
                        continue
                    stem = Path(member.name).stem.split("_")
                    if len(stem) < 3:
                        continue
                    speaker = "_".join(stem[:2])
                    if not re.fullmatch(r"[A-Za-z]+_[0-9]+", speaker):
                        continue
                    n = counts.get(speaker, 0)
                    if n >= args.clips_per_speaker:
                        continue
                    destination = target / speaker / f"{n:03d}.wav"
                    destination.parent.mkdir(exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    destination.write_bytes(source.read())
                    counts[speaker] = n + 1
                    records.append(
                        {
                            "speaker_id": speaker,
                            "path": str(destination.relative_to(ROOT)),
                            "archive_member": member.name,
                        }
                    )
    manifest = {
        "dataset": "NISP",
        "revision": REVISION,
        "group": args.group,
        "clips_per_speaker": args.clips_per_speaker,
        "speaker_count": len(counts),
        "recording_count": len(records),
        "records": records,
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    output = results / ("nisp_manifest_" + args.group.split("/")[0] + ".json")
    output.write_text(json.dumps(manifest, indent=2))
    print(
        f"Extracted {len(records)} clips from {len(counts)} speakers; {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
