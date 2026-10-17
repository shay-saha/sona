#!/usr/bin/env python3
"""Replay real audio through the application's persistent native wire protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import selectors
import struct
import subprocess
import time

import numpy as np
import soundfile as sf

from evaluate_conversation import CACHE, ROOT, reference, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/sona_stream')
    parser.add_argument('--wav', type=Path, default=CACHE / 'es2002a-live-60.wav')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--chunk-seconds', type=float, default=0.32)
    parser.add_argument('--output', type=Path, default=ROOT / 'research/results/streaming_replay.json')
    args = parser.parse_args()
    if not 0.05 <= args.chunk_seconds <= 5:
        parser.error('--chunk-seconds must be between 0.05 and 5')
    audio, rate = sf.read(args.wav, dtype='float32', always_2d=True)
    audio = audio.mean(axis=1)
    command = [str(args.binary), '--stream', '--rate', str(rate), '--models', str(CACHE)]
    if args.cpu:
        command.append('--cpu')
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    history, timings, updates = [], [], []

    def push(samples):
        began = time.perf_counter()
        process.stdin.write(struct.pack('<I', len(samples)) + samples.astype('<f4').tobytes())
        process.stdin.flush()
        if not selector.select(60):
            raise TimeoutError('Persistent native worker did not acknowledge audio')
        line = process.stdout.readline(16 * 1024 * 1024 + 1)
        if not line or len(line) > 16 * 1024 * 1024:
            raise ValueError('Missing or oversize native update')
        update = json.loads(line)
        if 'error' in update:
            raise ValueError(update['error'])
        if update.get('identity_space') != 'session':
            raise ValueError('Persistent worker returned window-local labels')
        timings.append(time.perf_counter() - began)
        offset = update['offset_seconds']
        prefix = [dict(segment, end=min(segment['end'], offset)) for segment in history
                  if segment['start'] < offset]
        history[:] = prefix + [dict(segment, start=segment['start'] + offset, end=segment['end'] + offset)
                               for segment in update['segments']]
        updates.append({'received_seconds': update['received_seconds'],
                        'analyzed_through': offset + update['duration_seconds'],
                        'acoustic_through': update.get('acoustic_through_seconds'),
                        'speakers': len({segment['speaker'] for segment in history})})
        return update

    try:
        began = time.perf_counter()
        chunk = max(1, round(rate * args.chunk_seconds))
        next_report = 0
        for offset in range(0, len(audio), chunk):
            end = min(len(audio), offset + chunk)
            if args.realtime:
                time.sleep(max(0, began + end / rate - time.perf_counter()))
            push(audio[offset:end])
            if offset >= next_report:
                print(f'Persistent replay: {offset / rate:.0f}/{len(audio) / rate:.0f} seconds', flush=True)
                next_report += rate * 10
        final = push(np.empty(0, dtype=np.float32))
        process.stdin.close()
        if process.wait(timeout=60) != 0:
            raise ValueError('Persistent worker failed at session end')
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    duration = len(audio) / rate
    if abs(final['offset_seconds'] + final['duration_seconds'] - duration) > 0.02:
        raise ValueError('Persistent replay lost its audio tail')
    report = {'schema_version': 1, 'audio_sha256': hashlib.sha256(args.wav.read_bytes()).hexdigest(),
              'duration_seconds': duration, 'updates': updates,
              'realtime_pacing': args.realtime, 'input_chunk_seconds': args.chunk_seconds,
              'device': final['device'],
              'push_wall_median_seconds': float(np.median(timings)),
              'push_wall_p95_seconds': float(np.percentile(timings, 95)),
              'maximum_push_wall_seconds': float(max(timings)),
              'maximum_live_push_wall_seconds': float(max(timings[:-1])),
              'final_flush_wall_seconds': timings[-1],
              'algorithmic_input_buffer_seconds': 1.12,
              'serial_replay_real_time_factor': float(sum(timings) / duration),
              'scope': 'Sequential pushes through the persistent native protocol with optional real-time pacing, '
                       'not a microphone capture or operating-system scheduling test. '
                       'Timing includes native inference and periodic acoustic completion. '
                       'One previously observed AMI development excerpt; no independent generalization claim.'}
    if args.wav.name == 'es2002a-live-60.wav':
        report['metrics'] = score(reference('ES2002a', 60, duration), history, duration)
        control = json.loads((ROOT / 'research/results/conversation_live.json').read_text())['cases'][0]
        report['context_control_der'] = control['der']
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'updates'}, indent=2))


if __name__ == '__main__':
    main()
