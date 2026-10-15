#!/usr/bin/env bash
set -euo pipefail

skip_download=false
reuse_features=false
for argument in "$@"; do
case "$argument" in
    --skip-download) skip_download=true ;;
    --reuse-features) reuse_features=true ;;
    --help|-h)
        cat <<'USAGE'
Usage: bash research/reproduce_physiology.sh [--skip-download] [--reuse-features]

Build the native tools and reproduce the expanded physiology and adult-age
experiments. The original pilot results remain frozen. Downloads total about
10 GB; audio, archives, encoder weights and the environment stay in research/cache.
Results and fitted regression parameters go to research/results.

Requires CMake 3.21+, C++17, Qt 6.4+, and Python 3.10+ with venv support.
Optional ONNX Runtime C++ headers/library enable age inference in the application.
--reuse-features reuses matching descriptor and encoder-output CSVs.
SONA_BUILD_DIR, SONA_JOBS and SONA_PYTHON override the usual build defaults.

NISP: https://github.com/iiscleap/NISP-Dataset
CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D (ODbL / DbCL)
Encoder: https://zenodo.org/records/7761387 (CC BY-NC-SA 4.0)
The encoder weights permit non-commercial use; they are downloaded, not bundled.
USAGE
        exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
esac
done

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=${SONA_BUILD_DIR:-"$project_root/build"}
research_python="$project_root/research/cache/venv/bin/python"
cmake -S "$project_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "${SONA_JOBS:-4}"
if [[ ! -x "$research_python" ]]; then
    "${SONA_PYTHON:-python3}" -m venv "$project_root/research/cache/venv"
fi
"$research_python" -m pip install -r "$project_root/research/requirements-age.txt"
if ! "$skip_download"; then
    for cohort in Hindi Kannada Malayalam Tamil Telugu; do
        "$research_python" "$project_root/research/scripts/fetch_nisp.py" --group "${cohort}_master/English_${cohort}" --workers 12
    done
    "$research_python" "$project_root/research/scripts/fetch_crema.py"
fi
"$research_python" "$project_root/research/scripts/fetch_age_encoder.py"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
feature_options=()
if "$reuse_features"; then feature_options+=(--reuse-features); fi
"$research_python" "$project_root/research/scripts/benchmark_physiology.py" --extractor "$build_dir/sona_features" "${feature_options[@]}"
"$research_python" "$project_root/research/scripts/benchmark_age.py" --extractor "$build_dir/sona_features" "${feature_options[@]}"
"$research_python" "$project_root/research/scripts/benchmark_noise.py" --extractor "$build_dir/sona_features" "${feature_options[@]}"
"$research_python" "$project_root/research/scripts/select_runtime.py"
"$research_python" "$project_root/research/scripts/audit_windows.py" --extractor "$build_dir/sona_features" "${feature_options[@]}"
"$research_python" "$project_root/research/scripts/verify_physiology.py" --build-dir "$build_dir"
