import gzip
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "fetch_nisp", Path(__file__).resolve().parents[1] / "scripts/fetch_nisp.py"
)
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class NispDownloadTests(unittest.TestCase):
    def test_same_size_corrupt_cache_is_replaced(self):
        payload = b"valid audio archive"
        item = {
            "path": "RECS.tar.gz.aa",
            "size": len(payload),
            "sha": git_blob_sha(payload),
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            destination = folder / item["path"]
            destination.write_bytes(bytes(len(payload)))
            with patch.object(
                fetch.urllib.request, "urlopen", return_value=io.BytesIO(payload)
            ):
                self.assertEqual(fetch.fetch_part(item, folder).read_bytes(), payload)

    def test_fresh_results_directory_and_double_gzip_archive(self):
        waveform = b"wav fixture"
        inner = io.BytesIO()
        with tarfile.open(fileobj=inner, mode="w:gz") as archive:
            entry = tarfile.TarInfo("RECS/Mal_0003_fixture.wav")
            entry.size = len(waveform)
            archive.addfile(entry, io.BytesIO(waveform))
        payload = gzip.compress(inner.getvalue())
        group = "Malayalam_master/English_Malayalam"
        item = {
            "path": group + "/RECS.tar.gz.aa",
            "size": len(payload),
            "sha": git_blob_sha(payload),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            parts = cache / group.replace("/", "_")
            parts.mkdir(parents=True)
            (parts / "RECS.tar.gz.aa").write_bytes(payload)
            (cache / "nisp_tree.json").write_text(
                json.dumps({"sha": fetch.REVISION, "tree": [item]})
            )

            def metadata_response(*args, **kwargs):
                return io.BytesIO(b"metadata fixture")

            with (
                patch.object(fetch, "ROOT", root),
                patch("sys.argv", ["fetch_nisp.py"]),
                patch.object(
                    fetch.urllib.request, "urlopen", side_effect=metadata_response
                ),
            ):
                fetch.main()
            manifest = json.loads(
                (root / "results/nisp_manifest_Malayalam_master.json").read_text()
            )
            self.assertEqual(manifest["speaker_count"], 1)
            self.assertEqual(manifest["recording_count"], 1)
            self.assertEqual(
                (root / manifest["records"][0]["path"]).read_bytes(), waveform
            )


if __name__ == "__main__":
    unittest.main()
