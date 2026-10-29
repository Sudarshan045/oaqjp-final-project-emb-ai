"""Detect emotions in text using the IBM Skills Network Watson NLP model."""

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyse):
    """Return five emotion scores and the name of the dominant emotion."""
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return None

    response = requests.post(
        URL,
        json={"raw_document": {"text": text_to_analyse}},
        headers=HEADERS,
        timeout=30,
    )
    if response.status_code == 400:
        return None
    response.raise_for_status()

    scores = response.json()["emotionPredictions"][0]["emotion"]
    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
