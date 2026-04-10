
from flask import Flask, render_template, Response, request, jsonify
import cv2, webbrowser

from modules.emotion_detection import detect_emotion
from modules.sentiment_analysis import analyze_text
from modules.context_detector import get_context
from modules.vibe_engine import map_vibe

app = Flask(__name__)

camera = cv2.VideoCapture(0)

current_emotion = "neutral"
current_vibe = "neutral"

playlist = {
"energetic":"https://music.youtube.com/search?q=energetic+songs",
"calm":"https://music.youtube.com/search?q=calm+music",
"relax":"https://music.youtube.com/search?q=relaxing+music",
"party":"https://music.youtube.com/search?q=party+songs",
"chill":"https://music.youtube.com/search?q=chill+music",
"neutral":"https://music.youtube.com/search?q=top+hits"
}

def gen_frames():
    global current_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, emotion = detect_emotion(frame)
        current_emotion = emotion

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/analyze_text", methods=["POST"])
def text_analysis():
    global current_vibe

    text = request.form.get("text")
    sentiment = analyze_text(text)
    context = get_context()

    vibe = map_vibe(current_emotion, sentiment, context)
    current_vibe = vibe

    return jsonify({
        "emotion": current_emotion,
        "sentiment": sentiment,
        "context": context,
        "vibe": vibe
    })

@app.route("/play")
def play():
    link = playlist.get(current_vibe, playlist["neutral"])
    webbrowser.open(link)
    return "opened"

if __name__ == "__main__":
    app.run(debug=True)
