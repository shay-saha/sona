#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
cuda=false
for argument in "$@"; do
    case "$argument" in
        --cuda) cuda=true ;;
        *) printf 'Usage: bash research/reproduce_streaming.sh [--cuda]\n' >&2; exit 2 ;;
    esac
done
research_python="$project_root/research/cache/venv/bin/python"
if [[ ! -x "$research_python" ]]; then
    python3 -m venv research/cache/venv
fi
"$research_python" -m pip install -r research/requirements.txt
fetch_options=(--jobs "${SONA_JOBS:-4}")
runtime_build="$project_root/research/cache/nemotron-build"
if "$cuda"; then
    fetch_options+=(--cuda)
    runtime_build="$project_root/research/cache/nemotron-cuda-build"
fi
"$research_python" research/scripts/fetch_streaming.py "${fetch_options[@]}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSONA_ENABLE_DIARIZATION=ON \
    -DSONA_NEMOTRON_ROOT="$runtime_build"
cmake --build build --parallel "${SONA_JOBS:-4}"
ctest --test-dir build -R 'conversation_|audio_source_transitions|gui_smoke' --output-on-failure
"$research_python" research/scripts/evaluate_conversation.py
"$research_python" research/scripts/benchmark_streaming.py
"$research_python" research/scripts/verify_streaming.py
if "$cuda"; then
    "$research_python" research/scripts/verify_streaming.py --gpu
fi
"$research_python" research/scripts/replay_conversation.py
"$research_python" research/scripts/replay_streaming.py --realtime
