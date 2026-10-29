"""Unit tests for the Watson emotion detector."""

import unittest
from unittest.mock import Mock, patch

from emotion_detection import EMOTIONS, emotion_detector


def response_for(dominant_emotion):
    """Build a deterministic Watson-shaped response for an offline test."""
    scores = {emotion: 0.01 for emotion in EMOTIONS}
    scores[dominant_emotion] = 0.95
    response = Mock(status_code=200)
    response.json.return_value = {
        "emotionPredictions": [{"emotion": scores}]
    }
    return response


class TestEmotionDetector(unittest.TestCase):
    @patch("emotion_detection.requests.post")
    def test_joy(self, post):
        post.return_value = response_for("joy")
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    @patch("emotion_detection.requests.post")
    def test_anger(self, post):
        post.return_value = response_for("anger")
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("emotion_detection.requests.post")
    def test_sadness(self, post):
        post.return_value = response_for("sadness")
        result = emotion_detector("I feel sad")
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("emotion_detection.requests.post")
    def test_fear(self, post):
        post.return_value = response_for("fear")
        result = emotion_detector("I am afraid")
        self.assertEqual(result["dominant_emotion"], "fear")

    @patch("emotion_detection.requests.post")
    def test_disgust(self, post):
        post.return_value = response_for("disgust")
        result = emotion_detector("That is disgusting")
        self.assertEqual(result["dominant_emotion"], "disgust")


if __name__ == "__main__":
    unittest.main()
