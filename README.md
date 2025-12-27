#  Breast Cancer Image Classification System

## Problem Statement

Early detection of breast cancer is crucial for effective treatment.
This project aims to classify breast cancer images as **Benign** or **Malignant** using machine learning and image processing techniques.

---

## Dataset

* Breast cancer histopathology images
* Organized into:

  ```
  data/
  ├── benign/
  └── malignant/
  ```
* Images are used to extract features for model training.

---

##  Technologies Used

* Python
* Flask
* OpenCV
* NumPy
* scikit-learn
* HTML, CSS

---

## Model / Approach

* Image feature extraction (intensity, texture, edges)
* **Random Forest Classifier** for prediction
* Model saved as `breast_cancer_simple_classifier.pkl`
* Flask used to integrate ML model with a web interface

---

##  Results

* The model classifies images into **Benign** or **Malignant**
* Displays prediction with a confidence score
* Designed for educational and demonstration purposes

---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/chandanakotha/breast-cancer-detection.git
cd breast-cancer-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open browser and visit:

```
http://127.0.0.1:5001
```

---

## 📂 Clean Folder Structure

```
Breast_Cancer_Detection/
├── app.py
├── data/
│   ├── benign/
│   └── malignant/
├── templates/
├── static/
├── uploads/
├── breast_cancer_simple_classifier.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📌 Git & Commits

Use **meaningful commits**, for example:

* `added image preprocessing`
* `trained random forest classifier`
* `integrated flask application`
* `added README and gitignore`

---

##  Disclaimer

This project is for **educational purposes only**.
It is **not intended for medical diagnosis**.

---

## 👩‍💻 Author

**Chandana Kotha**
Computer Science (AI & ML)
