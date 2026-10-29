"""Offline unit tests for the Watson emotion detector."""

import unittest
from unittest.mock import Mock, patch

from emotion_detection import EMOTIONS, HEADERS, URL, emotion_detector


class EmotionDetectorTests(unittest.TestCase):
    def setUp(self):
        self.scores = {
            "anger": 0.1,
            "disgust": 0.2,
            "fear": 0.3,
            "joy": 0.4,
            "sadness": 0.5,
        }
        self.payload = {"emotionPredictions": [{"emotion": self.scores}]}

    @patch("emotion_detection.requests.post")
    def test_scores_and_dominant_emotion(self, post):
        response = Mock(status_code=200)
        response.json.return_value = self.payload
        post.return_value = response

        result = emotion_detector("I feel something")

        self.assertEqual(
            result,
            {**self.scores, "dominant_emotion": "sadness"},
        )

    @patch("emotion_detection.requests.post")
    def test_request_uses_model_header_and_raw_document(self, post):
        response = Mock(status_code=200)
        response.json.return_value = self.payload
        post.return_value = response

        emotion_detector("I feel something")

        post.assert_called_once_with(
            URL,
            json={"raw_document": {"text": "I feel something"}},
            headers=HEADERS,
            timeout=30,
        )

    @patch("emotion_detection.requests.post")
    def test_result_contains_all_five_emotions(self, post):
        response = Mock(status_code=200)
        response.json.return_value = self.payload
        post.return_value = response

        result = emotion_detector("I feel something")

        self.assertEqual(set(EMOTIONS), set(result) - {"dominant_emotion"})

    @patch("emotion_detection.requests.post")
    def test_http_400_returns_none(self, post):
        post.return_value = Mock(status_code=400)

        self.assertIsNone(emotion_detector("invalid text"))

    @patch("emotion_detection.requests.post")
    def test_blank_input_returns_none_without_request(self, post):
        self.assertIsNone(emotion_detector("   "))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
