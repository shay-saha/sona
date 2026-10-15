#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python3 research/scripts/fetch_conversation.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSONA_ENABLE_DIARIZATION=ON
cmake --build build -j4
ctest --test-dir build -R 'conversation_|audio_source_transitions|gui_smoke' --output-on-failure
if [[ ! -x research/cache/venv/bin/python ]]; then
    python3 -m venv research/cache/venv
fi
research/cache/venv/bin/python -m pip install -r research/requirements.txt
research/cache/venv/bin/python research/scripts/evaluate_conversation.py
research/cache/venv/bin/python research/scripts/replay_conversation.py
