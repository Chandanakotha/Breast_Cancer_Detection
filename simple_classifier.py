#!/usr/bin/env python3
"""
Simple Image Feature-Based Classifier for Breast Cancer
"""

import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from datetime import datetime

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def extract_image_features(img_path, target_size=(64, 64)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []

        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.min(gray))
        features.append(np.max(gray))

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.append(np.mean(magnitude))
        features.append(np.std(magnitude))

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)

        features.append(np.std(grad_x))
        features.append(np.std(grad_y))

        texture_measure = np.var(gray)
        features.append(texture_measure)

        return np.array(features)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
def load_dataset(base_dir):
    print("Loading dataset...")

    benign_dir = os.path.join(base_dir, "data", "benign")
    malignant_dir = os.path.join(base_dir, "data", "malignant")

    X, y = [], []

    if os.path.exists(benign_dir):
        for file in os.listdir(benign_dir):
            if file.endswith(".png"):
                features = extract_image_features(os.path.join(benign_dir, file))
                if features is not None:
                    X.append(features)
                    y.append(0)

    if os.path.exists(malignant_dir):
        for file in os.listdir(malignant_dir):
            if file.endswith(".png"):
                features = extract_image_features(os.path.join(malignant_dir, file))
                if features is not None:
                    X.append(features)
                    y.append(1)

    X = np.array(X)
    y = np.array(y)

    print(f"Total samples: {len(X)}")
    print(f"Benign: {np.sum(y == 0)} | Malignant: {np.sum(y == 1)}")

    return X, y


# --------------------------------------------------
# Train model
# --------------------------------------------------
def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    return model


# --------------------------------------------------
# Save model
# --------------------------------------------------
def save_classifier(model, base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(base_dir, f"simple_classifier_{timestamp}.pkl")
    main_path = os.path.join(base_dir, "breast_cancer_simple_classifier.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(main_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved as: {main_path}")
    return main_path


# --------------------------------------------------
# Test model
# --------------------------------------------------
def test_classifier(model, base_dir):
    print("\nTesting model...")

    for label, folder in [("Benign", "benign"), ("Malignant", "malignant")]:
        dir_path = os.path.join(base_dir, "data", folder)
        if not os.path.exists(dir_path):
            continue

        files = [f for f in os.listdir(dir_path) if f.endswith(".png")][:3]
        print(f"\n{label} samples:")

        for file in files:
            features = extract_image_features(os.path.join(dir_path, file))
            if features is not None:
                proba = model.predict_proba([features])[0]
                pred = model.predict([features])[0]
                result = "Malignant" if pred == 1 else "Benign"
                print(f"{file} → {result} ({proba[pred]:.2f})")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Breast Cancer Image Classification (Simple ML)")
    print("=" * 60)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    X, y = load_dataset(BASE_DIR)
    if len(X) == 0:
        print("No data found!")
        return

    model = train_classifier(X, y)
    save_classifier(model, BASE_DIR)
    test_classifier(model, BASE_DIR)

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
