
from flask import Flask, render_template, Response, request, jsonify
import argparse
import os
import cv2
import webbrowser

from modules.emotion_detection import (
    DEEPFACE_ERROR,
    DeepFace,
    detect_emotion,
    personal_model,
)
from modules.sentiment_analysis import analyze_text
from modules.context_detector import get_context
from modules.vibe_engine import map_vibe

app = Flask(__name__)

def get_camera_index():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--camera", type=int)
    args, _ = parser.parse_known_args()

    if args.camera is not None:
        return args.camera

    return int(os.environ.get("CAMERA_INDEX", "0"))


def open_camera(index):
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends.insert(0, cv2.CAP_DSHOW)

    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if camera.isOpened():
            success, _ = camera.read()
            if success:
                return camera
        camera.release()

    raise RuntimeError(
        f"Could not open camera index {index}. Try another one, for example: "
        "python app.py --camera 1"
    )


camera = None
camera_index = get_camera_index()

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
    global camera, current_emotion

    if camera is None:
        camera = open_camera(camera_index)

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
    if personal_model is not None:
        print("Personal emotion model is available; using your calibrated model.")
    elif DeepFace is None:
        print("DeepFace is unavailable; using the basic OpenCV emotion fallback.")
        if DEEPFACE_ERROR is not None:
            print(f"DeepFace startup error: {DEEPFACE_ERROR}")
    else:
        print("DeepFace is available; using model-assisted emotion detection.")

    app.run(debug=True)
