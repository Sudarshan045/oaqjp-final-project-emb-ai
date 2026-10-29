"""Flask server for the Watson NLP emotion detector."""

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index():
    """Render the emotion detector form."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze submitted text or show the blank-input error."""
    text_to_analyse = request.args.get("textToAnalyze", "").strip()
    if not text_to_analyse:
        return render_template(
            "index.html",
            error="Invalid text! Please try again!",
        ), 400

    result = emotion_detector(text_to_analyse)
    if result is None or result["dominant_emotion"] is None:
        return render_template(
            "index.html",
            text=text_to_analyse,
            error="Invalid text! Please try again!",
        ), 400

    response_text = (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )
    return render_template(
        "index.html",
        text=text_to_analyse,
        result=result,
        response_text=response_text,
    )


if __name__ == "__main__":
    # Run static analysis with: python3 -m pylint server.py
    app.run(host="127.0.0.1", port=5000, debug=False)
