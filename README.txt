
Emotion-Based Vibe Music Recommender

Install dependencies:
pip install flask opencv-python textblob

For better emotion recognition, install the optional DeepFace model:
pip install deepface

Without DeepFace, the app uses a basic OpenCV fallback that can detect smiles
more reliably than anger or sadness.

If DeepFace or TensorFlow fails to start, the app will keep running with the
OpenCV fallback. TensorFlow compatibility depends on your Python version and
Windows setup.

To train a personal emotion model with your own webcam:
python collect_emotion_samples.py --camera 1

In the capture window, press:
h for happy
s for sad
a for angry
n for neutral
q to quit

Capture at least 25-40 samples for each emotion in similar lighting. Then train:
python train_personal_emotion_model.py

Restart the app after training. If models/personal_emotion_model.npz exists,
the app will use it before DeepFace or the basic OpenCV fallback.

Run:
python app.py

If Windows picks the wrong camera, list the OpenCV camera indexes:
python list_cameras.py

Then run with the onboard webcam index:
python app.py --camera 1

You can also set CAMERA_INDEX instead:
set CAMERA_INDEX=1
python app.py

Open browser:
http://127.0.0.1:5000
