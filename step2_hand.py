"""
Step 2 (finished): find your hand and draw its 21 points and the lines between them.

Run:  python step2_hand.py      (press Q in the video window to quit)
Needs hand_landmarker.task in the same folder.
"""
import time

import cv2
import mediapipe as mp

# Which points to join with a line. Each pair is (point number, point number).
# 0 = wrist, 4 = thumb tip, 8 = index tip, 12 = middle tip, 16 = ring tip, 20 = pinky tip
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# OpenCV colours are (Blue, Green, Red)
ORANGE = (35, 166, 246)
TEAL = (195, 214, 61)

# --- Set up the hand detector (configuration, written once) ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,   # we're feeding it a video, frame by frame
    num_hands=1,                      # look for one hand for now
)
landmarker = HandLandmarker.create_from_options(options)

# --- The step 1 loop, with detection added ---
camera = cv2.VideoCapture(0)
start = time.time()

while True:
    success, frame = camera.read()
    if not success:
        print("Couldn't read from the webcam")
        break
    frame = cv2.flip(frame, 1)

    # OpenCV stores colours as BGR, MediaPipe expects RGB, so convert
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # In video mode, MediaPipe needs each frame's time in milliseconds
    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # result.hand_landmarks is a list of hands; each hand is a list of 21 points
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        height, width, _ = frame.shape

        # Convert every point from 0–1 to pixels ONCE, and keep them in a list.
        # pixels[8] is then the index fingertip's (x, y) on screen.
        pixels = []
        for point in hand:
            pixels.append((int(point.x * width), int(point.y * height)))

        # Bonus: lines first, so the dots get drawn on top of them
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixels[a], pixels[b], TEAL, 3)

        # Dots on all 21 points
        for px, py in pixels:
            cv2.circle(frame, (px, py), 6, ORANGE, -1)

    cv2.putText(frame, "Press Q to quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("SignSetu", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()
