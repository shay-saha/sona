#!/usr/bin/env python3
"""Select neural speech thresholds on development, then audit fixed AMI integrations."""
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from evaluate_conversation import CACHE, ROOT, activity, reference, score

PRESETS = ['v3-streaming', 'v3-offline']
DEFAULTS = dict(onset=0.641, offset=0.561, padding_on=0.229, padding_off=0.079,
                minimum_duration=0.511, minimum_gap=0.296)


def segments(prediction, config):
    if prediction.get('first_frame', 0) != 0:
        raise ValueError('Development scoring requires the complete probability timeline')
    probabilities = np.asarray(prediction['frame_probabilities']).reshape(-1, prediction['speaker_capacity'])
    step = prediction['frame_seconds']
    duration = min(prediction['duration_seconds'], len(probabilities) * step)
    result = []
    for speaker, values in enumerate(probabilities.T):
        start = None
        spans = []
        for frame, value in enumerate(values):
            if start is None and value > config['onset']:
                start = frame * step
            elif start is not None and value < config['offset']:
                spans.append([max(0, start - config['padding_on']),
                              min(duration, frame * step + config['padding_off'])])
                start = None
        if start is not None:
            spans.append([max(0, start - config['padding_on']), duration])
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] < config['minimum_gap']:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        result.extend({'speaker': speaker, 'start': a, 'end': b} for a, b in merged
                      if b - a >= config['minimum_duration'])
    return result


def predictions(binary, meeting, microphone, preset, reuse):
    path = CACHE / f'nemotron-{meeting}-{microphone}-{preset}.json'
    if reuse and path.exists():
        return json.loads(path.read_text())
    clip = CACHE / f'{meeting}-{microphone}-60-240.wav'
    if not clip.exists():
        raise FileNotFoundError('Run evaluate_conversation.py to fetch the pinned AMI excerpts first')
    began = time.perf_counter()
    process = subprocess.run([str(binary), '--wav', str(clip), '--models', str(CACHE),
                              '--preset', preset, '--frames', '--neural-only', '--cpu'],
                             capture_output=True, text=True, check=True, timeout=600)
    output = json.loads(process.stdout)
    output['cpu_wall_seconds'] = time.perf_counter() - began
    path.write_text(json.dumps(output) + '\n')
    return output


def fuse(acoustic, neural, duration):
    if not neural or not acoustic:
        return []
    labels = sorted({segment['speaker'] for segment in neural})
    old, current = activity(acoustic, duration), activity(neural, duration)
    shared = old.astype(np.int64).T @ current.astype(np.int64)
    mapped = np.zeros_like(current)
    for cluster, channel in enumerate(shared.argmax(axis=1)):
        if shared[cluster, channel] >= 10:
            mapped[:, channel] |= old[:, cluster]
    assigned = np.where(current.any(axis=1)[:, None], current, mapped) & old.any(axis=1)[:, None]
    result = []
    for channel, label in enumerate(labels):
        edges = np.diff(np.r_[False, assigned[:, channel], False].astype(np.int8))
        result.extend({'speaker': label, 'start': a / 100, 'end': b / 100}
                      for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/sona_stream')
    parser.add_argument('--reuse-predictions', action='store_true')
    parser.add_argument('--development-only', action='store_true')
    args = parser.parse_args()
    baseline = json.loads((ROOT / 'research/results/conversation_ami.json').read_text())
    expected = reference('ES2002a', 60, 180)
    development = {preset: predictions(args.binary, 'ES2002a', 'Mix-Headset', preset,
                                       args.reuse_predictions) for preset in PRESETS}
    acoustic = json.loads((CACHE / 'ES2002a-Mix-Headset-60-240.prediction.json').read_text())['segments']
    grid = [DEFAULTS]
    for onset, fraction, minimum, padding in product([0.15, 0.3, 0.5], [0.6, 0.9], [0.1, 0.3], [0.03, 0.1]):
        grid.append(dict(onset=onset, offset=onset * fraction, minimum_duration=minimum,
                         padding_on=padding, padding_off=padding, minimum_gap=0.1))
    candidates = []
    for method in ['neural', 'fused']:
        for config in grid:
            metrics = {}
            for preset, prediction in development.items():
                proposed = segments(prediction, config)
                if method == 'fused':
                    proposed = fuse(acoustic, proposed, 180)
                metrics[preset] = score(expected, proposed, 180)
            candidates.append({'method': method, 'parameters': config, 'development': metrics,
                               'mean_der': float(np.mean([m['der'] for m in metrics.values()]))})
    control_error = abs(baseline['cases'][0]['estimated_speakers'] - baseline['cases'][0]['reference_speakers'])
    eligible = [row for row in candidates if all(abs(m['estimated_speakers'] - m['reference_speakers']) <= control_error
                                                for m in row['development'].values())]
    selected = min(eligible, key=lambda row: row['mean_der'])
    print(json.dumps(selected), flush=True)
    integrations = []
    if not args.development_only:
        for control in baseline['cases'][1:]:
            for preset in PRESETS:
                prediction = predictions(args.binary, control['meeting'], control['microphone'], preset,
                                         args.reuse_predictions)
                proposed = segments(prediction, selected['parameters'])
                if selected['method'] == 'fused':
                    old = CACHE / f"{control['meeting']}-{control['microphone']}-60-240.prediction.json"
                    proposed = fuse(json.loads(old.read_text())['segments'], proposed, 180)
                metrics = score(reference(control['meeting'], 60, 180), proposed, 180)
                result = {'meeting': control['meeting'], 'microphone': control['microphone'],
                          'preset': preset, 'control_der': control['der'],
                          'inference_push_seconds': float(sum(prediction['push_wall_seconds'])),
                          'maximum_push_wall_seconds': float(max(prediction['push_wall_seconds'])), **metrics}
                integrations.append(result)
                print(json.dumps(result), flush=True)
    development_pass = all(m['der'] < baseline['cases'][0]['der']
                           for m in selected['development'].values())
    integration_pass = bool(integrations) and all(case['der'] <= case['control_der'] + 0.02
                                                for case in integrations)
    if integrations:
        integration_pass &= np.mean([case['der'] for case in integrations]) < np.mean(
            [case['control_der'] for case in integrations])
    neural_default = next(row for row in candidates
                          if row['method'] == 'neural' and row['parameters'] == DEFAULTS)
    report = {'schema_version': 1, 'selected': selected, 'development_candidates': candidates,
              'ablation': {'acoustic_control': baseline['cases'][0],
                           'neural_with_upstream_segmentation': neural_default['development'],
                           'selected_fusion': selected['development']},
              'metric_gate': {'passes': bool(development_pass and integration_pass),
                              'native_silence_and_protocol_tests': 'tests/stream_protocol.py'},
              'integration': integrations,
              'protocol': json.loads((ROOT / 'research/experiments/streaming_voices.json').read_text()),
              'scope': 'Existing AMI integration clips, zero collar, overlap included. Both pretrained systems used AMI. '
                       'Processing runs through the native C ABI. This is an engineering comparison, not independent generalization.'}
    destination = ROOT / 'research/results/streaming_voices.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
