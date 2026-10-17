#!/usr/bin/env python3
"""Check deployed native fusion against frozen research predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmark_streaming import fuse, segments
from evaluate_conversation import CACHE, ROOT, activity, reference, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/sona_stream')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    selected = json.loads((ROOT / 'research/results/streaming_voices.json').read_text())['selected']
    cases = [('ES2002a', 'Mix-Headset'), ('IS1009a', 'Mix-Headset'), ('IS1009a', 'Array1-01')]
    results = []
    for meeting, microphone in cases:
        old = CACHE / f'{meeting}-{microphone}-60-240.prediction.json'
        neural = CACHE / f'nemotron-{meeting}-{microphone}-v3-offline.json'
        expected = fuse(json.loads(old.read_text())['segments'],
                        segments(json.loads(neural.read_text()), selected['parameters']), 180)
        clip = CACHE / f'{meeting}-{microphone}-60-240.wav'
        began = time.perf_counter()
        process = subprocess.run([str(args.binary), '--wav', str(clip), '--models', str(CACHE),
                                  '--preset', 'v3-offline', '--gpu' if args.gpu else '--cpu'],
                                 capture_output=True, text=True, check=True, timeout=600)
        actual = json.loads(process.stdout)
        elapsed = time.perf_counter() - began
        if actual['configuration']['method'] != 'neural_anchors_acoustic_completion':
            raise ValueError('Native worker did not deploy the selected method')
        a, b = activity(expected, 180), activity(actual['segments'], 180)
        if a.shape != b.shape:
            raise ValueError('Native speaker count disagrees with the research output')
        difference = float(np.any(a != b, axis=1).mean())
        if difference > (0.01 if args.gpu else 1e-6):
            raise ValueError(f'Native fusion differs from research: {meeting} {microphone}: {difference}')
        result = {'meeting': meeting, 'microphone': microphone, 'wall_seconds': elapsed,
                  'device': actual['configuration']['device'], 'different_frame_fraction': difference,
                  **score(reference(meeting, 60, 180), actual['segments'], 180)}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {'schema_version': 1, 'cases': results,
              'scope': 'Deployment parity on the same short AMI integration clips. GPU kernels may '
                       'differ slightly from the CPU reference near segmentation thresholds.'}
    suffix = 'gpu' if args.gpu else 'cpu'
    (ROOT / f'research/results/streaming_native_{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
