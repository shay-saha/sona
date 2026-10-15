#!/usr/bin/env bash
set -euo pipefail

skip_download=false
case "${1:-}" in
    --skip-download) skip_download=true ;;
    --help|-h)
        cat <<'USAGE'
Usage: bash research/reproduce.sh [--skip-download]

Build the native tools, fetch the pinned NISP cohort, extract descriptors,
train and evaluate the three regressors, run recording stress tests and
the confound audit, and verify exported parameters against native inference.

Requires CMake 3.21+, C++17, Qt 6.4+ (Widgets, Multimedia, OpenGL,
OpenGLWidgets, Concurrent), and Python 3.10+ with venv support.
The first run downloads approximately 1.63 GB. Cached audio and the Python
environment remain under research/cache; outputs go to research/results.

--skip-download requires an existing manifest, NISP metadata and extracted WAVs.
SONA_BUILD_DIR sets the build directory; SONA_JOBS sets parallel build jobs (4).
SONA_PYTHON selects the Python used to create the isolated environment.

Data: https://github.com/iiscleap/NISP-Dataset
Paper: https://arxiv.org/abs/2007.06021
Use the source's licence and intended-use terms when reusing its recordings.
USAGE
        exit 0
        ;;
    "") ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf 'Expected at most one argument.\n' >&2
    exit 2
fi

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${SONA_BUILD_DIR:-"$project_root/build"}
research_python="$project_root/research/cache/venv/bin/python"

cmake -S "$project_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "${SONA_JOBS:-4}"
if [[ ! -x "$research_python" ]]; then
    "${SONA_PYTHON:-python3}" -m venv "$project_root/research/cache/venv"
fi
"$research_python" -m pip install -r "$project_root/research/requirements.txt"
if ! "$skip_download"; then
    "$research_python" "$project_root/research/scripts/fetch_nisp.py"
fi

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$research_python" "$project_root/research/scripts/benchmark_nisp.py" --extractor "$build_dir/sona_features"
"$research_python" "$project_root/research/scripts/stress_nisp.py" --extractor "$build_dir/sona_features"
"$research_python" "$project_root/research/scripts/audit_nisp.py"
"$research_python" "$project_root/research/scripts/verify_native.py" --build-dir "$build_dir"
