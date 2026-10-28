"""Offline unit tests for the Watson emotion detector."""

import unittest
from unittest.mock import Mock, patch

from emotion_detection import URL, emotion_detector


class EmotionDetectorTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.2,
                    "fear": 0.3,
                    "joy": 0.4,
                    "sadness": 0.5,
                }
            }]
        }

    @patch("emotion_detection.requests.post")
    def test_scores_and_dominant_emotion(self, post):
        response = Mock(status_code=200)
        response.json.return_value = self.payload
        post.return_value = response
        self.assertEqual(emotion_detector("I feel something"), {
            "anger": 0.1,
            "disgust": 0.2,
            "fear": 0.3,
            "joy": 0.4,
            "sadness": 0.5,
            "dominant_emotion": "sadness",
        })
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], URL)

    @patch("emotion_detection.requests.post")
    def test_http_400_returns_none(self, post):
        post.return_value = Mock(status_code=400)
        self.assertIsNone(emotion_detector("bad input"))

    @patch("emotion_detection.requests.post")
    def test_blank_input_returns_none_without_request(self, post):
        self.assertIsNone(emotion_detector("  "))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
