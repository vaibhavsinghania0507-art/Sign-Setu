# SignSetu

Turns hand signs into text and speech using only a laptop webcam. You can teach it a new sign in under a minute.

*Setu* means bridge.

## Setup

1. Install Python 3.9 or later.
2. Install the libraries:
   ```
   python -m pip install -r requirements.txt
   ```
3. Download MediaPipe's hand model into this folder:
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Run

```
python signsetu.py
```

Click the video window, then:

| Key | What it does |
|---|---|
| T | Teach a new sign. Type its name in the terminal, then hold the sign steady for the countdown and recording |
| C | Clear the sentence |
| Q | Quit |

Your examples are saved in `signs.csv`, so the app remembers signs between runs.

To measure accuracy:

```
python evaluate.py
```

## How it works

1. **OpenCV** reads frames from the webcam.
2. **MediaPipe Hand Landmarker** finds 21 points on the hand in each frame.
3. Each point is measured relative to the wrist and divided by the palm size, so only the hand's shape matters, not its position or distance from the camera.
4. A **k-nearest neighbours** classifier (scikit-learn) compares the hand with the recorded examples. If the hand is far from every example, or the neighbours disagree, it doesn't guess.
5. A sign only counts once it's been the same for 12 frames in a row, which stops flickering.
6. The word is added to the sentence on screen and spoken with **pyttsx3** (works offline).

## Limitations

- Tracks one hand, and only still hand shapes, not moving signs.
- Trained on whoever records the examples, so it works best for their hand.

## Files

- `signsetu.py`: the app
- `evaluate.py`: measures accuracy on `signs.csv`
- `step2_hand.py`: a simpler version that just draws the hand, useful for understanding the basics
