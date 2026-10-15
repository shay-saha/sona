import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_conversation import score


def segment(speaker, start, end):
    return {'speaker': speaker, 'start': start, 'end': end}


class ConversationScoringTests(unittest.TestCase):
    def test_permuted_labels_have_no_error(self):
        reference = [segment(0, 0, 2), segment(1, 1, 3)]
        hypothesis = [segment(9, 0, 2), segment(4, 1, 3)]
        result = score(reference, hypothesis, 4)
        self.assertEqual(result['der'], 0)
        self.assertEqual(result['overlap_precision'], 1)
        self.assertEqual(result['overlap_recall'], 1)

    def test_missing_overlap_counts_speaker_time(self):
        reference = [segment(0, 0, 2), segment(1, 1, 3)]
        result = score(reference, [segment(7, 0, 2)], 4)
        self.assertEqual(result['der'], 0.5)
        self.assertEqual(result['miss_rate'], 0.5)
        self.assertEqual(result['confusion_rate'], 0)

    def test_wrong_voice_has_confusion(self):
        reference = [segment(0, 0, 2), segment(1, 2, 4)]
        result = score(reference, [segment(7, 0, 4)], 4)
        self.assertEqual(result['der'], 0.5)
        self.assertEqual(result['miss_rate'], 0)
        self.assertEqual(result['confusion_rate'], 0.5)

    def test_background_activity_counts_false_alarm(self):
        result = score([segment(0, 0, 1)], [segment(0, 0, 2)], 3)
        self.assertEqual(result['false_alarm_rate'], 1)
        self.assertEqual(result['der'], 1)


if __name__ == '__main__':
    unittest.main()
