#!/usr/bin/env python3
"""Compare the frozen height controls and verify restoration in native inference."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT.parent / 'build/sona_traits')
    args = parser.parse_args()
    original = json.loads((ROOT / 'results/nisp_pilot_models.json').read_text())['models'][0]
    replacement = json.loads((ROOT / 'results/nisp_noise_models.json').read_text())['models'][0]
    restored = json.loads((ROOT / 'results/nisp_runtime_models.json').read_text())['models'][0]
    labels = pd.read_csv(ROOT / 'cache/nisp_total_spkrinfo.list', sep=r'\s+').set_index('Speaker_ID').Height
    results = {}
    for condition in ['clean', 'white_noise_15db']:
        frame = pd.read_csv(ROOT / f'results/windows_test_{condition}_features.csv')
        frame['speaker_id'] = frame.file.map(lambda p: Path(p).stem.rsplit('_', 1)[0])
        frame = frame.loc[frame.file.str.endswith('_10s.wav') &
                          ~frame.speaker_id.str.startswith('Mal_') &
                          (frame.active_fraction >= 0.3) & (frame.periodic_fraction >= 0.2) &
                          (frame.clipped_fraction <= 0.01)]
        truth = labels.loc[frame.speaker_id].to_numpy()
        result = {'speakers': len(frame), 'reference_height_sd_cm': float(truth.std())}
        for name, model in [('original', original), ('replacement', replacement)]:
            x = (frame[model['feature_names']].to_numpy() - model['means']) / model['scales']
            prediction = x @ np.asarray(model['weights']) + model['intercept']
            result[name] = {'mae_cm': float(abs(prediction - truth).mean()),
                            'prediction_sd_cm': float(prediction.std()),
                            'slope_against_height': float(np.polyfit(truth, prediction, 1)[0])}
        results[condition] = result
    errors = []
    for path in sorted((ROOT / 'cache/nisp_audio/Mal_0003').glob('*.wav')):
        predictions = []
        for model in ['nisp_pilot_models.json', 'nisp_runtime_models.json']:
            output = subprocess.check_output([str(args.binary), str(ROOT / 'results' / model), str(path)], text=True)
            predictions.append(json.loads(output)['traits'][0]['estimate'])
        if all(value is not None for value in predictions):
            errors.append(abs(predictions[0] - predictions[1]))
    assert errors and max(errors) < 1e-7
    assert original['weights'] == restored['weights'][:len(original['weights'])]
    report = {'schema_version': 1, 'windows': results,
              'native_original_parity': {'comparisons': len(errors), 'maximum_absolute_difference_cm': max(errors)},
              'decision': 'Restore original coefficients for behavior compatibility; retain replacement as an explicitly loadable comparison.',
              'scope': 'Descriptive audit on existing 10-second windows. Malayalam excluded because some identities were fitted by the original model. No new model fitting or claim that rollback improves population MAE.'}
    (ROOT / 'results/height_control_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
