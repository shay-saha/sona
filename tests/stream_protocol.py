import array
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary, models = sys.argv[1:]
command = [binary, '--models', models, '--cpu']
silence = array.array('f', [0]) * (16000 * 12)
batch = subprocess.run([*command, '--stdin', '--rate', '16000'], input=silence.tobytes(),
                        capture_output=True, check=True, timeout=60)
result = json.loads(batch.stdout)
assert result['segments'] == [] and result['duration_seconds'] == 12
assert result['configuration']['method'] == 'neural_anchors_acoustic_completion'
chunk = array.array('f', [0]) * (24000 * 3)
packet = struct.pack('<I', len(chunk)) + chunk.tobytes()
replay = subprocess.run([*command, '--stream', '--rate', '24000'], input=packet * 4 + struct.pack('<I', 0),
                         capture_output=True, check=True, timeout=60)
updates = [json.loads(line) for line in replay.stdout.splitlines()]
assert len(updates) == 5 and updates[-1]['final']
assert all(update['segments'] == [] and update['identity_space'] == 'session' for update in updates)
assert updates[-1]['received_seconds'] == 12
assert abs(updates[-1]['offset_seconds'] + updates[-1]['duration_seconds'] - 12) < 0.001
through = [update['offset_seconds'] + update['duration_seconds'] for update in updates]
assert through == sorted(through), 'Native stream time must advance monotonically'
for malformed in [b'\x01\x00', struct.pack('<I', 4 * 1024 * 1024 + 1), struct.pack('<I', 20) + b'\x00']:
    process = subprocess.run([*command, '--stream'], input=malformed, capture_output=True, timeout=20)
    assert process.returncode != 0 and 'error' in json.loads(process.stdout)
with tempfile.TemporaryDirectory() as directory:
    Path(directory, 'Nemotron-3-Diarization.q8_0.gguf').write_bytes(b'invalid')
    process = subprocess.run([binary, '--models', directory, '--stream', '--cpu'],
                             input=b'', capture_output=True, timeout=20)
    assert process.returncode != 0 and 'checksum' in json.loads(process.stdout)['error']
print('Neural batch/stream silence, resampling, framing and checksum guards passed')
