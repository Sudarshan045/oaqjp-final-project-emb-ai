# Final Project

## Project name

Emotion Detector

This project analyzes English text with the IBM Skills Network Watson NLP emotion model and displays scores for anger, disgust, fear, joy, and sadness in a Flask web application.

## Run the application

```bash
python3 -m pip install -r requirements.txt
python3 test_emotion_detection.py
python3 server.py
```

Open `http://127.0.0.1:5000` in a browser. The Watson endpoint must be reachable for live emotion scores.

## Package use

```python
from EmotionDetection.emotion_detection import emotion_detector

print(emotion_detector("I am so happy I am doing this"))
```
