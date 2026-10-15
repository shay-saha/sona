#!/usr/bin/env python3
"""Score native anonymous diarization against AMI manual word/vocal annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'research/cache/conversation'
ANNOTATIONS = 'https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip'
ANNOTATION_HASH = 'b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d'
CASES = [('ES2002a', 'Mix-Headset', 'development'), ('IS1009a', 'Mix-Headset', 'integration'),
         ('IS1009a', 'Array1-01', 'integration')]


def reference(meeting, start, duration):
    segments = []
    with zipfile.ZipFile(CACHE / 'ami_public_manual_1.6.2.zip') as archive:
        for speaker, channel in enumerate('ABCD'):
            root = ET.fromstring(archive.read(f'words/{meeting}.{channel}.words.xml'))
            for element in root:
                if element.tag not in ('w', 'vocalsound'):
                    continue
                a = max(0.0, float(element.get('starttime', 0)) - start)
                b = min(duration, float(element.get('endtime', 0)) - start)
                if b > a:
                    segments.append({'speaker': speaker, 'start': a, 'end': b})
    return segments


def activity(segments, duration):
    labels = sorted({s['speaker'] for s in segments})
    index = {s: i for i, s in enumerate(labels)}
    grid = (np.arange(round(duration * 100)) + 0.5) / 100
    matrix = np.zeros((len(grid), len(labels)), dtype=bool)
    for s in segments:
        matrix[:, index[s['speaker']]] |= (grid >= s['start']) & (grid < s['end'])
    return matrix


def score(expected, predicted, duration):
    ref, hyp = activity(expected, duration), activity(predicted, duration)
    overlap = ref.astype(np.int64).T @ hyp.astype(np.int64)
    rows, cols = linear_sum_assignment(-overlap)
    correct = np.zeros(ref.shape[0])
    for r, c in zip(rows, cols):
        correct += ref[:, r] & hyp[:, c]
    nr, nh = ref.sum(1), hyp.sum(1)
    denominator = nr.sum()
    miss, false = np.maximum(nr - nh, 0).sum(), np.maximum(nh - nr, 0).sum()
    confusion = (np.minimum(nr, nh) - correct).sum()
    ro, ho = nr >= 2, nh >= 2
    matched_time_error = [abs(ref[:, r].sum() - hyp[:, c].sum()) / 100 for r, c in zip(rows, cols)]
    return {'der': float((miss + false + confusion) / max(1, denominator)),
            'miss_rate': float(miss / max(1, denominator)),
            'false_alarm_rate': float(false / max(1, denominator)),
            'confusion_rate': float(confusion / max(1, denominator)),
            'reference_speakers': ref.shape[1], 'estimated_speakers': hyp.shape[1],
            'reference_overlap_seconds': float(ro.sum() / 100),
            'estimated_overlap_seconds': float(ho.sum() / 100),
            'overlap_precision': float((ro & ho).sum() / max(1, ho.sum())),
            'overlap_recall': float((ro & ho).sum() / max(1, ro.sum())),
            'matched_speaking_time_mae_seconds': float(np.mean(matched_time_error)) if matched_time_error else None}


def download(url, path, expected=None):
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/sona_diarize')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/results/conversation_ami.json')
    parser.add_argument('--reuse-predictions', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    download(ANNOTATIONS, CACHE / 'ami_public_manual_1.6.2.zip', ANNOTATION_HASH)
    results = []
    for meeting, microphone, role in CASES:
        filename = f'{meeting}.{microphone}.wav'
        url = f'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{filename}'
        digest = download(url, CACHE / filename)
        start, duration = 60, 180
        with sf.SoundFile(CACHE / filename) as audio:
            audio.seek(start * audio.samplerate)
            samples = audio.read(duration * audio.samplerate, dtype='float32', always_2d=True).mean(axis=1)
            rate = audio.samplerate
        clip = CACHE / f'{meeting}-{microphone}-60-240.wav'
        sf.write(clip, samples, rate, subtype='PCM_16')
        prediction_path = clip.with_suffix('.prediction.json')
        began = time.perf_counter()
        if args.reuse_predictions and prediction_path.exists():
            prediction = json.loads(prediction_path.read_text())
            elapsed = None
        else:
            process = subprocess.run([str(args.binary), '--wav', str(clip), '--models', str(CACHE)],
                                     capture_output=True, text=True, timeout=600, check=True)
            prediction = json.loads(process.stdout)
            prediction_path.write_text(json.dumps(prediction))
            elapsed = time.perf_counter() - began
        if abs(prediction['duration_seconds'] - duration) > 0.01:
            raise ValueError('Worker did not analyze the entire clip')
        result = {'meeting': meeting, 'microphone': microphone, 'role': role,
                  'source_url': url, 'source_sha256': digest, 'start_seconds': start, 'duration_seconds': duration,
                  'cpu_wall_seconds': elapsed, **score(reference(meeting, start, duration),
                                                      prediction['segments'], duration)}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {'schema_version': 1, 'backend': prediction['backend'],
              'configuration': prediction.get('configuration', {}), 'cases': results,
              'protocol': {'reference': 'AMI manual 1.6.2 words and vocal sounds; zero-duration items excluded',
                           'annotation_url': ANNOTATIONS, 'annotation_sha256': ANNOTATION_HASH,
                           'frame_seconds': 0.01, 'collar_seconds': 0, 'overlap_included': True,
                           'mapping': 'Maximum global speaker-time overlap using Hungarian assignment',
                           'selection': 'Encoder/clustering settings inspected only on ES2002a 60–240 s, then frozen',
                           'scope': 'Three short integration checks from two AMI meetings, not a corpus benchmark; '
                                    'segmentation was pretrained using AMI; no claim of independent generalization',
                           'limitations': 'No calibrated identity confidence; unknown speaker count and overlap can fail. '
                                          'Continuous activity spans are counted as turns; they are not linguistic utterances.'}}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
