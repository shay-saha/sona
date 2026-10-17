#!/usr/bin/env python3
"""Build pinned native speaker tracking dependencies and verify the model download."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from fetch_conversation import fetch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_REVISION = 'caa3f47c7c75fa2e9e6340272ebad57c1647dd44'
SENTENCEPIECE_REVISION = '31646a467d2051eb904e0b45de3a73e91fe1c1e3'
MODEL_REVISION = 'f667ed73aee57d40cc39428eb768b4fd87a0a29e'
MODEL_HASH = '08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1'
MODEL_NAME = 'Nemotron-3-Diarization.q8_0.gguf'


def run(*command):
    subprocess.run([str(arg) for arg in command], check=True)


def checkout(url, path, revision):
    if not path.exists():
        run('git', 'init', path)
        run('git', '-C', path, 'remote', 'add', 'origin', url)
        run('git', '-C', path, 'fetch', '--depth', '1', 'origin', revision)
        run('git', '-C', path, 'checkout', '--detach', revision)
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise ValueError(f'Unexpected source revision in {path}: {actual}')
    run('git', '-C', path, 'diff', '--exit-code', '--quiet')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'research/cache')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cuda-architecture', default='native')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 64:
        parser.error('--jobs must be between 1 and 64')
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    run(sys.executable, ROOT / 'research/scripts/fetch_conversation.py', '--cache', cache / 'conversation')
    fetch(f'https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/{MODEL_REVISION}/{MODEL_NAME}',
          cache / 'conversation' / MODEL_NAME, MODEL_HASH)
    sentencepiece = cache / 'sentencepiece-source'
    checkout('https://github.com/google/sentencepiece.git', sentencepiece, SENTENCEPIECE_REVISION)
    sp_build = cache / 'sentencepiece-build'
    run('cmake', '-S', sentencepiece, '-B', sp_build, '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON', '-DSPM_ENABLE_SHARED=OFF',
        '-DSPM_ENABLE_TCMALLOC=OFF', '-DSPM_BUILD_TEST=OFF')
    run('cmake', '--build', sp_build, '--target', 'sentencepiece-static', '--parallel', args.jobs)
    source = cache / 'nemotron-runtime'
    checkout('https://github.com/NVIDIA/NeMo-Speech.cpp.git', source, RUNTIME_REVISION)
    run('git', '-C', source, 'submodule', 'update', '--init', '--depth', '1', 'llama.cpp')
    build = cache / ('nemotron-cuda-build' if args.cuda else 'nemotron-build')
    options = ['-DCMAKE_BUILD_TYPE=Release', '-DNEMO_SPEECH_BUILD_ASR=OFF',
               '-DNEMO_SPEECH_BUILD_TTS=OFF', '-DNEMO_SPEECH_BUILD_CLI=OFF',
               '-DNEMO_SPEECH_BUILD_EXAMPLES=OFF', '-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF',
               '-DBUILD_TESTING=OFF', f'-DGGML_CUDA={"ON" if args.cuda else "OFF"}',
               f'-DSENTENCEPIECE_STATIC_LIB={sp_build}/src/libsentencepiece.a',
               f'-DSENTENCEPIECE_INCLUDE_DIR={sentencepiece}/src']
    if args.cuda:
        options.append(f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_architecture}')
    run('cmake', '-S', source, '-B', build, *options)
    run('cmake', '--build', build, '--target', 'nemo_speech_asr_c', '--parallel', args.jobs)
    print(shlex.join(['cmake', '-S', '.', '-B', 'build', f'-DSONA_NEMOTRON_ROOT={build}']))
    print('cmake --build build -j4')


if __name__ == '__main__':
    main()
