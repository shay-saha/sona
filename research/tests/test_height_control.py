import json
from pathlib import Path
import unittest

RESULTS = Path(__file__).resolve().parents[1] / 'results'


class OriginalHeightControlTests(unittest.TestCase):
    def test_original_height_parameters_are_preserved(self):
        original = json.loads((RESULTS / 'nisp_pilot_models.json').read_text())['models'][0]
        runtime = json.loads((RESULTS / 'nisp_runtime_models.json').read_text())['models'][0]
        self.assertEqual(runtime['source_model'], 'nisp_pilot_models.json')
        self.assertEqual(runtime['intercept'], original['intercept'])
        for key in ['means', 'scales', 'weights']:
            self.assertEqual(runtime[key][:46], original[key])
        self.assertTrue(all(value == 0 for value in runtime['weights'][46:]))

    def test_original_training_cohort_is_excluded_from_new_calibration(self):
        report = json.loads((RESULTS / 'live_window_report.json').read_text())['targets']['height_cm']
        self.assertFalse(any(name.startswith('Mal_') for name in report['interval_calibration_ids']))
        self.assertFalse(any(name.startswith('Mal_') for name in report['evaluation_ids']))


if __name__ == '__main__':
    unittest.main()
