"""Emotion detector packaged for EmotionDetection imports."""

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
NO_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def emotion_detector(text_to_analyse):
    """Return the five emotion scores and the dominant emotion."""
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return NO_RESULT.copy()

    try:
        response = requests.post(
            URL,
            json={"raw_document": {"text": text_to_analyse}},
            headers=HEADERS,
            timeout=30,
        )
        if response.status_code == 400:
            return NO_RESULT.copy()
        response.raise_for_status()

        scores = response.json()["emotionPredictions"][0]["emotion"]
        result = {emotion: scores[emotion] for emotion in EMOTIONS}
        result["dominant_emotion"] = max(result, key=result.get)
        return result
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return NO_RESULT.copy()
