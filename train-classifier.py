import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from joints import FEATURE_COLS

DATASET = "labeled_dataset.csv"
OUTPUT  = "stroke_classifier.pkl"

# Load
if not os.path.exists(DATASET):
    print(f"[!] {DATASET} not found — run classify.py first to label some strokes")
    exit()

print("Loading dataset...")
df = pd.read_csv(DATASET)

n_fh    = len(df[df["label"] == "forehand_drive"])
n_other = len(df[df["label"] == "other"])
print(f"  Total  : {len(df)} frames")
print(f"  forehand_drive : {n_fh}")
print(f"  other          : {n_other}")

if n_fh < 30:
    print("[!] Not enough forehand_drive frames — label more strokes and try again")
    exit()

# Features and labels
X = df[FEATURE_COLS].values
y = df["label"].values

# 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining on {len(X_train)} frames, testing on {len(X_test)}...")

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("\n--- Accuracy ---")
print(classification_report(y_test, y_pred))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=["forehand_drive", "other"])
print(f"                 Predicted FH    Predicted Other")
print(f"  Actual FH      {cm[0][0]:<16} {cm[0][1]}")
print(f"  Actual Other   {cm[1][0]:<16} {cm[1][1]}")

# Feature importance
print("\n--- Top 5 Most Important Joints ---")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(min(5, len(FEATURE_COLS))):
    idx = indices[i]
    print(f"  {FEATURE_COLS[idx]:<25} {importances[idx]:.4f}")

# Save
joblib.dump(model, OUTPUT)
print(f"\n[✓] Saved to {OUTPUT}")
print("Now run main.py — it will load the classifier automatically")