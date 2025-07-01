# Facial Recognition System for Secure Employee Identification 🧠📸

A deep learning-based face verification system designed to authenticate employee identities securely using facial embeddings and binary classification models.

> 👨‍💻 **Developer:** Abhishek Gupta  
> 🎓 **Academic Major Project | 2025**  
> 🌐 **Deployed on Streamlit Cloud**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Demo](#demo)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Future Scope](#future-scope)
- [Credits](#credits)

---

## ✅ Overview

This project demonstrates a facial recognition system that can identify if a given face matches a known identity — specifically trained to classify between **Arnold Schwarzenegger** and **non-Arnold faces** using **128-dimensional embeddings** and a deep neural network.

---

## ❓ Problem Statement

Traditional ID systems are manual, error-prone, and insecure. This project aims to:
- Automate identity verification
- Eliminate manual errors
- Enable real-time, secure employee authentication using face embeddings

---

## 📂 Dataset

- **Source:** Derived from Labeled Faces in the Wild (LFW)
- **Used File:** `lfw_arnie_nonarnie.csv`
- **Description:**  
  - Each row is a 128-dimensional vector representing a face embedding  
  - Label `1` → Arnold Schwarzenegger  
  - Label `0` → Other identities  

---

## ⚙️ Methodology

1. **Data Loading & Cleaning**
2. **Preprocessing:**  
   - Scaled embeddings using `StandardScaler`  
   - Encoded labels (1 = Arnold, 0 = Not)
3. **Model Training:**
   - Deep neural network built using Keras
   - 2 hidden layers with dropout
4. **Evaluation:**  
   - Test accuracy, confusion matrix
5. **Deployment:**  
   - Streamlit app with simulated image input

---

## 🧠 Model Architecture

```text
Input Layer: 128 neurons (face embedding)
Hidden Layer 1: Dense(128), ReLU, Dropout(0.4)
Hidden Layer 2: Dense(64), ReLU, Dropout(0.3)
Output Layer: Dense(1), Sigmoid
Loss Function: Binary Crossentropy
Optimizer: Adam
Epochs: 50 (with early stopping)
📈 Results
Metric	Value
Test Accuracy	78.95%
Model Type	Binary Classifier (Arnold vs Not)
Evaluation	Confusion Matrix, Prediction Confidence

The model successfully learned the embedding space and performed well given limited classes.

💻 Demo
🟢 Streamlit Web App (Deployed):
👉 Live App Link

Features:
Upload face image (simulated embedding)

Returns prediction: Arnold ✅ / Not ❌

Displays confidence score

🛠 How to Run Locally
🔧 Clone the repo:
bash
Copy
Edit
git clone https://github.com/A289shek2004/Facial-Recognition-System-for-Secure-Employee-Identification-project.git
cd Facial-Recognition-System-for-Secure-Employee-Identification-project
💾 Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
🚀 Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
📦 Files Included
File	Description
app.py	Streamlit UI app
requirements.txt	Python dependencies
lfw_arnie_nonarnie.csv	Face embedding dataset
face_identification_model.h5	Trained Keras model
scaler.pkl	Fitted StandardScaler
README.md	Project documentation

🧰 Technologies Used
Python 🐍

Pandas & NumPy

Scikit-learn

TensorFlow / Keras

Streamlit

Matplotlib / Seaborn

StandardScaler, LabelEncoder

🚀 Future Scope
Real-time webcam support via OpenCV

Liveness detection (anti-spoofing)

Multiple class detection (not just Arnold)

Integration with employee attendance systems

Deployment on Raspberry Pi or mobile apps

🙏 Credits
Labeled Faces in the Wild (LFW Dataset)

FaceNet/Dlib for embedding generation

Streamlit Cloud for deployment

👋 Connect With Me
📧 Email: abhishekgup2004@gmail.com

🔗 LinkedIn: linkedin.com/in/abhishek-gupta-56a8ab286

💻 GitHub: github.com/A289shek2004

“AI is not just about automation — it’s about augmenting human potential.” – Abhishek Gupta

yaml
Copy
Edit

---

### ✅ Final Step

Save this content as `README.md` and place it in your project folder.

Then commit:

```bash
git add README.md
git commit -m "📝 Re-added complete README with full project workflow"
git push
