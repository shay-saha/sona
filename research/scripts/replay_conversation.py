#!/usr/bin/env python3
"""Compare native rolling conversation windows on a fixed annotated AMI excerpt."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf

from evaluate_conversation import ANNOTATIONS, ANNOTATION_HASH, CACHE, ROOT, download, reference, score

CONFIGURATIONS = [('context-control', 30, 5), ('short-context', 15, 2), ('faster-updates', 30, 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/sona_diarize')
    parser.add_argument('--reuse-predictions', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/results/conversation_live.json')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    download(ANNOTATIONS, CACHE / 'ami_public_manual_1.6.2.zip', ANNOTATION_HASH)
    baseline = json.loads((ROOT / 'research/results/conversation_ami.json').read_text())['cases'][0]
    source = CACHE / 'ES2002a.Mix-Headset.wav'
    download(baseline['source_url'], source, baseline['source_sha256'])
    with sf.SoundFile(source) as audio:
        audio.seek(60 * audio.samplerate)
        samples = audio.read(60 * audio.samplerate, dtype='float32', always_2d=True).mean(axis=1)
        rate = audio.samplerate
    clip = CACHE / 'es2002a-live-60.wav'
    sf.write(clip, samples, rate, subtype='PCM_16')
    expected = reference('ES2002a', 60, 60)
    results = []
    for name, window, step in CONFIGURATIONS:
        path = CACHE / f'es2002a-live-{name}.json'
        if args.reuse_predictions and path.exists():
            prediction = json.loads(path.read_text())
        else:
            process = subprocess.run([str(args.binary), '--wav', str(clip), '--models', str(CACHE),
                                      '--rolling-window', str(window), '--rolling-step', str(step)],
                                     text=True, capture_output=True, check=True, timeout=600)
            prediction = json.loads(process.stdout)
            path.write_text(json.dumps(prediction) + '\n')
        if abs(prediction['duration_seconds'] - 60) > 0.01:
            raise ValueError('Incomplete rolling replay')
        timings = prediction['window_wall_seconds']
        result = {'name': name, 'window_seconds': window, 'step_seconds': step,
                  'configuration': prediction['configuration'],
                  'window_wall_median_seconds': float(np.median(timings)),
                  'window_wall_p95_seconds': float(np.percentile(timings, 95)),
                  **score(expected, prediction['segments'], 60)}
        results.append(result)
        print(json.dumps(result), flush=True)
    control_count_error = abs(results[0]['estimated_speakers'] - results[0]['reference_speakers'])
    eligible = [case for case in results
                if abs(case['estimated_speakers'] - case['reference_speakers']) <= control_count_error]
    preferred = min(eligible, key=lambda case: case['der'])
    report = {'schema_version': 1, 'cases': results,
              'preferred_configuration': preferred['name'],
              'selection_rule': 'Lowest DER among configurations that do not increase speaker-count error relative to context control.',
              'source_url': baseline['source_url'], 'source_sha256': baseline['source_sha256'],
              'excerpt_start_seconds': 60, 'duration_seconds': 60,
              'protocol': 'Native worker followed by the same ConversationHistory used in the application. '
                          'Unspecified speaker count, 10 ms scoring, no collar, overlap included, global Hungarian mapping.',
              'scope': 'One previously observed AMI development excerpt. An engineering comparison, '
                       'not independent validation or a corpus benchmark. All configurations use current identity matching.',
              'timing_limit': 'CPU timings depend on hardware and contention. Replay advances audio time '
                              'after each completed request; it does not simulate microphone scheduling or dropped updates.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
