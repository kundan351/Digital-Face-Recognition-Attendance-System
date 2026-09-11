import os
import cv2
import numpy as np
import pickle
import urllib.request
from sklearn.ensemble import RandomForestClassifier

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Use ABSOLUTE paths so files are always found/saved in the same place,
# regardless of the working directory the app happens to be started from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")

# --- Face detector model (MediaPipe Tasks API) ---
# Newer MediaPipe releases removed the old `mediapipe.solutions` API
# entirely (it now raises `AttributeError: module 'mediapipe' has no
# attribute 'solutions'`). The replacement is the "MediaPipe Tasks" API,
# which needs an explicit .tflite model file instead of a built-in one.
FACE_DETECTOR_MODEL_PATH = os.path.join(APP_DIR, "blaze_face_short_range.tflite")
FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

_face_detector = None


def _ensure_face_detector_model():
    """Download the .tflite face detector model on first use if it isn't
    already present. Requires the machine running the Flask app to have
    internet access. If it doesn't, download the file manually with:

        wget -O blaze_face_short_range.tflite \
            https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

    and place it next to model.py.
    """
    if os.path.exists(FACE_DETECTOR_MODEL_PATH):
        return
    try:
        urllib.request.urlretrieve(FACE_DETECTOR_MODEL_URL, FACE_DETECTOR_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the face detector model automatically "
            f"({e}). Please download it manually:\n"
            f"wget -O {FACE_DETECTOR_MODEL_PATH} {FACE_DETECTOR_MODEL_URL}"
        )


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _ensure_face_detector_model()
        base_options = mp_python.BaseOptions(model_asset_path=FACE_DETECTOR_MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options, min_detection_confidence=0.5
        )
        _face_detector = mp_vision.FaceDetector.create_from_options(options)
    return _face_detector


def _detect_faces_bgr(bgr_image):
    detector = _get_face_detector()
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    return result.detections


# ---- Utility: extract face crop -> small grayscale vector (embedding) ----
def crop_face_and_embed(bgr_image, detection):
    h, w = bgr_image.shape[:2]
    # NOTE: the new Tasks API returns bounding_box in ABSOLUTE pixel
    # coordinates (origin_x, origin_y, width, height), unlike the old
    # `solutions` API which returned fractional (0-1) coordinates.
    bbox = detection.bounding_box
    x1 = max(0, bbox.origin_x)
    y1 = max(0, bbox.origin_y)
    x2 = min(w, bbox.origin_x + bbox.width)
    y2 = min(h, bbox.origin_y + bbox.height)
    if x2 <= x1 or y2 <= y1:
        return None
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb


def extract_embedding_for_image(stream_or_bytes):
    # accepts a file-like stream (werkzeug FileStorage.stream)
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    detections = _detect_faces_bgr(img)
    if not detections:
        return None
    return crop_face_and_embed(img, detections[0])


# ---- Load model helpers ----
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_with_model(clf, emb):
    # returns label and confidence (max probability)
    proba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf


# ---- Training function used in background ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    dataset_dir/
        student_id/
            img1.jpg
            img2.jpg

    progress_callback(progress_percent, message, done=False) -> optional.
    `done=True` MUST be passed on the final call (success, failure, or
    "no data") so the caller knows training has actually stopped and can
    flip its "running" flag back off.
    """
    X = []
    y = []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_students = max(1, len(student_dirs))
    processed = 0

    for sid in student_dirs:
        folder = os.path.join(dataset_dir, sid)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            detections = _detect_faces_bgr(img)
            if not detections:
                continue
            emb = crop_face_and_embed(img, detections[0])
            if emb is None:
                continue
            X.append(emb)
            y.append(int(sid))
        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)  # up to 80% during feature extraction
            progress_callback(pct, f"Processed {processed}/{total_students} students", False)

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found (no faces detected in any uploaded image)", True)
        return

    if len(set(y)) < 2:
        # RandomForest will technically still "fit" with a single class, but
        # it will then always predict that one student no matter who is in
        # front of the camera, which looks like a broken model. Fail loudly
        # instead so the user knows to enroll a second student first.
        if progress_callback:
            progress_callback(0, "Need at least 2 students with valid face images to train", True)
        return

    X = np.stack(X)
    y = np.array(y)

    if progress_callback:
        progress_callback(85, "Training RandomForest...", False)
    clf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    clf.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    if progress_callback:
        progress_callback(100, "Training complete", True)