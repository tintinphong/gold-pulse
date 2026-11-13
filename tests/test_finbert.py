"""Offline tests for the FinBERT path: no model download, no torch needed."""
import unittest

import numpy as np
import pandas as pd

from processing.bucket import aggregate_hourly, apply_sentiment_file
from sentiment.finbert import probs_to_frame
from tests.test_bucket import T0, posts

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}  # ProsusAI/finbert's class order


class ProbsToFrameTests(unittest.TestCase):
    def test_score_is_pos_minus_neg_and_label_is_most_likely_class(self):
        probs = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.2, 0.5]])
        f = probs_to_frame(probs, ID2LABEL)
        np.testing.assert_allclose(f["finbert_score"], [0.6, -0.5, 0.1])
        self.assertEqual(f["finbert_label"].tolist(), ["bullish", "bearish", "neutral"])
        np.testing.assert_allclose(f["finbert_neu"], [0.2, 0.3, 0.5])


class SentimentFileTests(unittest.TestCase):
    def setUp(self):
        self.posts = posts((1, "a", 0.9, 1, 0), (2, "b", -0.9, 1, 0), (3, "c", 0.0, 1, 0))
        self.scores = pd.DataFrame({"post_id": ["p0", "p1", "p2"],
                                    "finbert_score": [0.02, -0.4, 0.3],
                                    "finbert_label": ["neutral", "bearish", "bullish"]})

    def test_scores_and_labels_replace_vader(self):
        out = apply_sentiment_file(self.posts, self.scores)
        self.assertEqual(out["sentiment"].tolist(), [0.02, -0.4, 0.3])
        hour = aggregate_hourly(out).loc[T0]
        self.assertAlmostEqual(hour["mean_sentiment"], (0.02 - 0.4 + 0.3) / 3)
        # ratios follow FinBERT's labels, not the +/-0.05 band (0.02 would be neutral either way,
        # so check a post the band and the label disagree on)
        self.assertAlmostEqual(hour["bullish_ratio"], 1 / 3)
        self.assertAlmostEqual(hour["bearish_ratio"], 1 / 3)

    def test_label_overrides_band(self):
        scores = self.scores.assign(finbert_score=[0.3, 0.3, 0.3],
                                    finbert_label=["neutral", "neutral", "neutral"])
        hour = aggregate_hourly(apply_sentiment_file(self.posts, scores)).loc[T0]
        self.assertEqual(hour["bullish_ratio"], 0)

    def test_unscored_post_is_an_error_not_a_silent_mix(self):
        with self.assertRaises(ValueError):
            apply_sentiment_file(self.posts, self.scores.iloc[:2])

    def test_without_file_vader_band_is_unchanged(self):
        hour = aggregate_hourly(self.posts).loc[T0]
        self.assertAlmostEqual(hour["bullish_ratio"], 1 / 3)
        self.assertAlmostEqual(hour["mean_sentiment"], 0.0)


if __name__ == "__main__":
    unittest.main()
