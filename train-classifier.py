import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn import tree
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

# Show counts for every label in the dataset dynamically
labels = df["label"].unique().tolist()
print(f"  Total : {len(df)} frames")
for l in labels:
    print(f"  {l} : {len(df[df['label'] == l])}")

# Warn if any stroke class is too small but don't block training
for l in labels:
    if l != "other" and len(df[df["label"] == l]) < 30:
        print(f"[!] Warning: only {len(df[df['label'] == l])} frames for '{l}' — accuracy may be low")

# Features and labels
X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
y = df["label"].to_numpy()

# 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining on {len(X_train)} frames, testing on {len(X_test)}...")

# Train
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=10, class_weight="balanced")
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("\n--- Accuracy ---")
print(classification_report(y_test, y_pred))

# Confusion matrix — dynamic based on whatever classes exist
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred, labels=labels)
header = f"{'':20}" + "".join(f"{l:20}" for l in labels)
print(header)
for i, row_label in enumerate(labels):
    row = f"  {row_label:18}" + "".join(f"{cm[i][j]:<20}" for j in range(len(labels)))
    print(row)

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