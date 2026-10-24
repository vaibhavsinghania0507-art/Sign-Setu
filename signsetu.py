"""
SignSetu: turns hand signs into text and speech, using only a webcam.

Run:  python signsetu.py
Needs hand_landmarker.task in the same folder.

Controls (click the video window first):
    T   teach a new sign (you type its name in the terminal)
    C   clear the sentence
    Q   quit

How it works, in one line per step:
    1. OpenCV reads frames from the webcam.
    2. MediaPipe finds the 21 points of your hand in each frame.
    3. The points are made position- and size-independent (hand_to_features).
    4. A k-nearest neighbours model compares them with the examples in signs.csv.
    5. A sign only counts once it's the same for several frames in a row.
    6. The word is added to the sentence on screen and spoken aloud.
"""
import csv
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------
MODEL_FILE = "hand_landmarker.task"   # MediaPipe's hand model
DATA_FILE = "signs.csv"               # your recorded examples live here

SAMPLES_PER_SIGN = 60    # frames recorded each time you teach a sign (~2 seconds)
COUNTDOWN_SECONDS = 3    # time to get your hand in position before recording
STABLE_FRAMES = 12       # a sign must stay the same for this many frames to count
K_NEIGHBOURS = 5         # how many nearest examples the model looks at
MIN_CONFIDENCE = 0.8     # at least 4 of the 5 neighbours must agree
MAX_DISTANCE = 1.2       # if the hand is this far from every example, it's "unknown".
                         # Raise it if real signs get ignored, lower it if random
                         # hand shapes get recognised as signs.

# Which points to join with a line (0 = wrist, 4/8/12/16/20 = fingertips)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# OpenCV colours are (Blue, Green, Red)
ORANGE = (35, 166, 246)
TEAL = (195, 214, 61)
WHITE = (255, 255, 255)
DARK = (38, 23, 14)


# ---------------------------------------------------------------------------
# 1–2. Seeing the hand
# ---------------------------------------------------------------------------
def create_hand_detector():
    """Load MediaPipe's hand model, set up for video."""
    if not os.path.exists(MODEL_FILE):
        raise SystemExit(f"Can't find {MODEL_FILE}. Download it into this folder first.")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_FILE),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def find_hand(detector, frame, timestamp_ms):
    """Return the 21 points of the first hand in the frame, or None if there's no hand."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # MediaPipe wants RGB
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, timestamp_ms)
    if result.hand_landmarks:
        return result.hand_landmarks[0]
    return None


def to_pixels(hand, frame):
    """Turn MediaPipe's 0–1 positions into pixel positions on this frame."""
    height, width, _ = frame.shape
    return [(int(p.x * width), int(p.y * height)) for p in hand]


def draw_hand(frame, pixels):
    """Draw the lines between joints, then a dot on every point."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pixels[a], pixels[b], TEAL, 3)
    for px, py in pixels:
        cv2.circle(frame, (px, py), 6, ORANGE, -1)


# ---------------------------------------------------------------------------
# 3. Making the data fair
# ---------------------------------------------------------------------------
def hand_to_features(hand, frame):
    """
    Turn 21 points into 42 numbers that describe the hand's SHAPE only.

    - Subtract the wrist, so the wrist is always at (0, 0): moving your hand
      around the screen no longer changes the numbers.
    - Divide by the palm size (wrist to the base of the middle finger, point 9):
      moving your hand closer to the camera no longer changes the numbers.
    """
    height, width, _ = frame.shape
    # Use real pixel distances so the picture isn't stretched (frames aren't square)
    points = np.array([(p.x * width, p.y * height) for p in hand], dtype=float)

    points = points - points[0]                 # wrist becomes (0, 0)
    palm_size = np.linalg.norm(points[9])       # length of the palm
    if palm_size == 0:
        return None
    points = points / palm_size

    return points.flatten()                     # 21 (x, y) pairs -> 42 numbers


# ---------------------------------------------------------------------------
# 3–4. The dataset and the model
# ---------------------------------------------------------------------------
def load_dataset():
    """Read signs.csv. Each row is: sign name, then 42 numbers."""
    features, labels = [], []
    if not os.path.exists(DATA_FILE):
        return features, labels
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            labels.append(row[0])
            features.append([float(value) for value in row[1:]])
    return features, labels


def save_samples(name, samples):
    """Add newly recorded examples to the end of signs.csv."""
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sample in samples:
            writer.writerow([name] + [f"{value:.5f}" for value in sample])


def train_model():
    """Train k-nearest neighbours on everything in signs.csv. Returns None if it's empty."""
    features, labels = load_dataset()
    if not labels:
        return None
    k = min(K_NEIGHBOURS, len(labels))          # can't look at 5 neighbours if we only have 3 examples
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(features, labels)
    print(f"Model trained on {len(labels)} examples of {len(set(labels))} signs: "
          f"{', '.join(sorted(set(labels)))}")
    return model


def predict_sign(model, features):
    """Return the sign the model thinks this is, or None if it isn't sure."""
    # How far away are the closest examples? Far away = a hand shape it has never seen.
    distances, _ = model.kneighbors([features])
    if distances.mean() > MAX_DISTANCE:
        return None

    # What fraction of the nearest neighbours vote for each sign?
    votes = model.predict_proba([features])[0]
    best = votes.argmax()
    if votes[best] < MIN_CONFIDENCE:
        return None
    return model.classes_[best]


# ---------------------------------------------------------------------------
# 6. Speaking
# ---------------------------------------------------------------------------
def speak(word):
    """Say the word out loud. The video pauses for a moment while it talks."""
    try:
        engine = pyttsx3.init()      # a fresh engine each time avoids pyttsx3 going silent
        engine.say(word)
        engine.runAndWait()
        engine.stop()
    except Exception as error:       # speech failing shouldn't crash the whole app
        print(f"Couldn't speak '{word}': {error}")


# ---------------------------------------------------------------------------
# Drawing text on screen
# ---------------------------------------------------------------------------
def draw_label(frame, text, position, scale=0.7, colour=WHITE, thickness=2):
    """Draw text with a dark box behind it so it's readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - 8, y - text_h - 8), (x + text_w + 8, y + baseline + 4), DARK, -1)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_centre_text(frame, text, scale=1.4):
    height, width, _ = frame.shape
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    draw_label(frame, text, ((width - text_w) // 2, height // 2), scale, ORANGE, 3)


# ---------------------------------------------------------------------------
# The main loop
# ---------------------------------------------------------------------------
def main():
    detector = create_hand_detector()
    model = train_model()
    if model is None:
        print("No signs yet. Press T in the video window to teach your first one.")

    camera = cv2.VideoCapture(0)
    start = time.time()

    # The app is always in one of three modes:
    #   "recognise" - normal use
    #   "countdown" - 3, 2, 1 before recording a new sign
    #   "recording" - saving examples of a new sign
    mode = "recognise"
    new_sign_name = ""
    countdown_end = 0.0
    recorded = []

    recent = deque(maxlen=STABLE_FRAMES)   # the last few predictions, for the flicker fix
    sentence = []                          # words recognised so far
    last_added = None                      # stops one held sign being added again and again

    while True:
        success, frame = camera.read()
        if not success:
            print("Couldn't read from the webcam")
            break
        frame = cv2.flip(frame, 1)

        timestamp_ms = int((time.time() - start) * 1000)
        hand = find_hand(detector, frame, timestamp_ms)

        features = None
        if hand is not None:
            draw_hand(frame, to_pixels(hand, frame))
            features = hand_to_features(hand, frame)

        # ---------------- Teaching: countdown ----------------
        if mode == "countdown":
            seconds_left = countdown_end - time.time()
            if seconds_left > 0:
                draw_centre_text(frame, f"Show '{new_sign_name}' in {math.ceil(seconds_left)}")
            else:
                mode = "recording"
                recorded = []

        # ---------------- Teaching: recording ----------------
        elif mode == "recording":
            if features is not None:           # only keep frames where a hand was found
                recorded.append(features)
            draw_centre_text(frame, f"Recording '{new_sign_name}' "
                                    f"{len(recorded)}/{SAMPLES_PER_SIGN}")
            if hand is None:
                draw_label(frame, "Can't see your hand!", (20, 80), colour=ORANGE)

            if len(recorded) >= SAMPLES_PER_SIGN:
                save_samples(new_sign_name, recorded)
                model = train_model()           # retrain straight away with the new sign
                mode = "recognise"
                recent.clear()
                last_added = None

        # ---------------- Normal use: recognising ----------------
        else:
            word = None
            if model is not None and features is not None:
                word = predict_sign(model, features)
            recent.append(word)

            if word is not None:
                draw_label(frame, f"Seeing: {word}", (20, 80), colour=ORANGE)

            # Flicker fix: only accept a sign once every recent frame agrees
            buffer_full = len(recent) == STABLE_FRAMES
            all_same = len(set(recent)) == 1
            if buffer_full and all_same:
                if word is not None and word != last_added:
                    sentence.append(word)
                    last_added = word
                    speak(word)
                elif word is None:
                    last_added = None           # hand gone or unknown: the same sign can count again

        # ---------------- Always on screen ----------------
        height, _, _ = frame.shape
        draw_label(frame, "T: teach a sign   C: clear   Q: quit", (20, 40), scale=0.6)
        if sentence:
            draw_label(frame, " ".join(sentence[-6:]), (20, height - 30), scale=1.0)

        cv2.imshow("SignSetu", frame)

        # ---------------- Keys ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
            last_added = None
        elif key == ord('t') and mode == "recognise":
            # Tell the user to look at the terminal (the window freezes while input() waits)
            draw_centre_text(frame, "Type the sign's name in the terminal", scale=0.9)
            cv2.imshow("SignSetu", frame)
            cv2.waitKey(1)
            name = input("Name of the new sign (then press Enter): ").strip()
            if name:
                new_sign_name = name
                countdown_end = time.time() + COUNTDOWN_SECONDS
                mode = "countdown"
                print("Click back on the video window and hold the sign steady.")

    camera.release()
    cv2.destroyAllWindows()
    detector.close()


# Only start the app when this file is run directly
# (so evaluate.py can import load_dataset without opening the webcam)
if __name__ == "__main__":
    main()
