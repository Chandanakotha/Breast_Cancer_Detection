from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import requests
import pickle
import cv2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# Feature Extraction
# ===============================
def extract_image_features(img_path, target_size=(64, 64)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.resize(img, target_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = [
            np.mean(gray),
            np.std(gray),
            np.min(gray),
            np.max(gray)
        ]

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.sum(cv2.Canny(gray, 50, 150) > 0) / (gray.shape[0] * gray.shape[1]),
            np.std(grad_x),
            np.std(grad_y),
            np.var(gray)
        ])

        return np.array(features)

    except Exception as e:
        print(f"Image processing error: {e}")
        return None


# ===============================
# Load Classifier
# ===============================
classifier_path = os.path.join(BASE_DIR, "breast_cancer_simple_classifier.pkl")
classifier = None

try:
    with open(classifier_path, "rb") as f:
        classifier = pickle.load(f)
    print("✅ Classifier loaded successfully")
except Exception as e:
    print("❌ Classifier load failed:", e)


# ===============================
# Test images (NEW DATA PATH)
# ===============================
test_image_path = os.path.join(
    BASE_DIR, "data", "malignant", "10253_idx5_x451_y651_class0.png"
)

test_benign_path = os.path.join(
    BASE_DIR, "data", "benign", "10253_idx5_x351_y1851_class0.png"
)


# ===============================
# Flask App
# ===============================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence = "", ""

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            prediction = "No file selected"
        else:
            uploads_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            filepath = os.path.join(uploads_dir, file.filename)
            file.save(filepath)

            features = extract_image_features(filepath)
            if features is not None and classifier:
                proba = classifier.predict_proba([features])[0]
                pred = classifier.predict([features])[0]
                prediction = "Malignant" if pred == 1 else "Benign"
                confidence = f"{max(0.6, min(0.99, proba[pred])):.1%}"
            else:
                prediction = "Error processing image"

    return render_template("index.html", prediction=prediction, confidence=confidence)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/feature_analysis")
def feature_analysis():
    return render_template("feature_analysis.html")


@app.route("/image_analysis", methods=["GET", "POST"])
def image_analysis():
    prediction, confidence = "", ""

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            prediction = "No file selected"
        else:
            uploads_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            filepath = os.path.join(uploads_dir, file.filename)
            file.save(filepath)

            features = extract_image_features(filepath)
            if features is not None and classifier:
                proba = classifier.predict_proba([features])[0]
                pred = classifier.predict([features])[0]
                prediction = "Malignant" if pred == 1 else "Benign"
                confidence = f"{max(0.6, min(0.99, proba[pred])):.1%}"
            else:
                prediction = "Error processing image"

    return render_template("image_analysis.html", prediction=prediction, confidence=confidence)


@app.route("/predict_features", methods=["POST"])
def predict_features():
    try:
        features = [float(request.form.get(f"feature_{i}")) for i in range(4)]
        feature_prediction = "Malignant" if sum(features) > 50 else "Benign"
    except:
        feature_prediction = "Invalid input"

    return render_template("feature_analysis.html", feature_prediction=feature_prediction)


@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    age = int(request.form.get("age", 0))
    brca1 = request.form.get("brca1", "")
    relatives = request.form.get("relatives", "")

    if age > 50 or brca1 == "positive" or relatives == "more":
        risk_prediction = "High risk"
    else:
        risk_prediction = "Low / Moderate risk"

    return render_template("analysis.html", risk_prediction=risk_prediction)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    if not GROQ_API_KEY:
        return jsonify({"reply": "AI service not configured."})

    user_message = request.get_json().get("message", "")

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": "You are a responsible medical assistant."},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 300
            },
            timeout=15
        )
        reply = response.json()["choices"][0]["message"]["content"]
    except:
        reply = "AI service unavailable."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
