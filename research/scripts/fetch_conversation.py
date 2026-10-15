#!/usr/bin/env python3
"""Fetch the pinned CPU diarization runtime and models into the ignored cache."""
import argparse
import hashlib
from pathlib import Path
import platform
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'cache' / 'conversation'
VERSION = '1.13.8'
RUNTIME = f'sherpa-onnx-v{VERSION}-linux-x64-shared-no-tts-lib'
FILES = [
    (f'https://github.com/k2-fsa/sherpa-onnx/releases/download/v{VERSION}/{RUNTIME}.tar.bz2',
     'runtime.tar.bz2', 'bf2d998c8b07012cd5098f3b92673bc1333fd9b927767d7cb664be8190d8bc0b'),
    ('https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2',
     'segmentation.tar.bz2', '24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488'),
    ('https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx',
     'nemo_en_titanet_small.onnx', 'ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e'),
    (f'https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v{VERSION}/sherpa-onnx/c-api/c-api.h',
     'include/sherpa-onnx/c-api/c-api.h', '2a1b95084be8fd1deb3228fcad2fd3f7f0258b64582f7402281ec174c7b7f4ce'),
]


def fetch(url, path, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        temporary = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {path.name}', flush=True)
        urllib.request.urlretrieve(url, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
            temporary.unlink()
            raise ValueError(f'Checksum mismatch: {path.name}')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT)
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('The pinned prebuilt runtime is Linux x86_64; use a sherpa-onnx 1.13.8 source build on other platforms.')
    for url, relative, digest in FILES:
        fetch(url, args.cache / relative, digest)
    for name, sentinel, digest in [
        ('runtime.tar.bz2', f'{RUNTIME}/lib/libsherpa-onnx-c-api.so',
         'c067b18cd746e49eb5016ee428196d21e1df14c14a3c54f128c1ac95f6aaed50'),
        ('segmentation.tar.bz2', 'sherpa-onnx-pyannote-segmentation-3-0/model.onnx',
         '220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079'),
    ]:
        target = args.cache / sentinel
        if target.exists() and (hashlib.sha256(target.read_bytes()).hexdigest() == digest):
            continue
        with tarfile.open(args.cache / name) as archive:
            archive.extractall(args.cache, filter='data')
    print(f'cmake -S . -B build -DSONA_SHERPA_ROOT={args.cache / RUNTIME}')
    print('cmake --build build -j4')


if __name__ == '__main__':
    main()
