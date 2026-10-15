import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary, models = sys.argv[1:]
samples = array.array('f', [0]) * (16000 * 12)
process = subprocess.run([binary, '--stdin', '--rate', '16000', '--models', models],
                         input=samples.tobytes(), capture_output=True, timeout=60, check=True)
result = json.loads(process.stdout)
assert result['duration_seconds'] == 12, 'Pipe transfers must retain all audio, including multiple read buffers'
assert result['segments'] == [] and result['speech_seconds'] == 0, 'Silence must not create anonymous people'
process = subprocess.run([binary, '--stdin', '--rate', '16000', '--models', models,
                          '--rolling-window', '15', '--rolling-step', '2'],
                         input=samples.tobytes(), capture_output=True, timeout=60, check=True)
replay = json.loads(process.stdout)
assert replay['duration_seconds'] == 12 and replay['segments'] == []
assert len(replay['window_wall_seconds']) == 5, 'Replay must include its shorter final update'
long_samples = array.array('f', [0]) * (16000 * 301)
process = subprocess.run([binary, '--stdin', '--rate', '16000', '--models', models],
                         input=long_samples.tobytes(), capture_output=True, timeout=60, check=True)
long_result = json.loads(process.stdout)
assert long_result['duration_seconds'] == 301 and long_result['silence_seconds'] == 301
assert long_result['segments'] == [], 'Chunked recordings must retain complete coverage without inventing people'
with tempfile.TemporaryDirectory() as root:
    Path(root, 'sherpa-onnx-pyannote-segmentation-3-0').mkdir()
    Path(root, 'sherpa-onnx-pyannote-segmentation-3-0/model.onnx').write_bytes(b'invalid')
    process = subprocess.run([binary, '--stdin', '--rate', '16000', '--models', root],
                             input=samples.tobytes(), capture_output=True, timeout=10)
    assert process.returncode != 0 and 'checksum' in json.loads(process.stdout)['error']
print('Native worker accepts complete audio and rejects corrupt models')
