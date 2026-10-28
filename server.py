"""Flask web interface for the Watson NLP emotion detector."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/emotionDetector")
def detect_emotion():
    text_to_analyze = request.form.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return render_template(
            "index.html", error="Please enter some text to analyze."
        ), 400

    result = emotion_detector(text_to_analyze)
    if result is None:
        return render_template(
            "index.html", error="The text could not be analyzed. Please try again."
        ), 400
    return render_template("index.html", text=text_to_analyze, result=result)


if __name__ == "__main__":
    # Static analysis command: python3 -m pylint server.py
    app.run(host="127.0.0.1", port=5000, debug=False)
