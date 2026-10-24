"""
Measure how accurate SignSetu is on your recorded signs.

Run:  python evaluate.py

For each sign, the model trains on the first 80% of its examples and is tested
on the last 20%. We split by time order instead of randomly, because frames
recorded next to each other are almost identical. A random split would let
the model "cheat" and give you an accuracy that's too good to be true.

For the most honest number: teach each sign a second time on a different day
(or with a friend's hand). Those later recordings end up in the test set.
"""
from collections import defaultdict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from signsetu import K_NEIGHBOURS, load_dataset

TEST_FRACTION = 0.2

features, labels = load_dataset()
if not labels:
    raise SystemExit("signs.csv is empty. Teach some signs in signsetu.py first.")

# Group the rows by sign, keeping them in the order they were recorded
rows_by_sign = defaultdict(list)
for feature, label in zip(features, labels):
    rows_by_sign[label].append(feature)

train_x, train_y, test_x, test_y = [], [], [], []
for sign, rows in rows_by_sign.items():
    split = int(len(rows) * (1 - TEST_FRACTION))
    train_x += rows[:split]
    train_y += [sign] * split
    test_x += rows[split:]
    test_y += [sign] * (len(rows) - split)

if len(rows_by_sign) < 2:
    print("Note: with only one sign, accuracy will always be 100%. Teach a few more.")

model = KNeighborsClassifier(n_neighbors=min(K_NEIGHBOURS, len(train_y)))
model.fit(train_x, train_y)
predictions = model.predict(test_x)

print(f"\nSigns: {len(rows_by_sign)}   Training examples: {len(train_y)}   "
      f"Test examples: {len(test_y)}")
print(f"Accuracy: {accuracy_score(test_y, predictions) * 100:.1f}%\n")

print("Per sign:")
print(classification_report(test_y, predictions, zero_division=0))

# Rows = the real sign, columns = what the model guessed.
# Numbers off the diagonal show which signs get mixed up.
names = sorted(rows_by_sign)
matrix = confusion_matrix(test_y, predictions, labels=names)
width = max(len(n) for n in names) + 2
print("Confusion matrix (row = real sign, column = guess):")
print(" " * width + "".join(n[:8].rjust(9) for n in names))
for name, row in zip(names, matrix):
    print(name.ljust(width) + "".join(str(v).rjust(9) for v in row))
