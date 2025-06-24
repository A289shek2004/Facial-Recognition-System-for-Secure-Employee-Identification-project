# Facial Recognition System for Secure Employee Identification

A Deep Learning-based facial verification system built using face embeddings and neural networks to securely verify employee identities in real-time.

> 🎓 Major Project 2  
> 👨‍💻 Developed by: Abhishek Gupta  (Internship project)
> 📁 Dataset: Precomputed Face Embeddings (LFW Subset)

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Demo](#demo)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [User Interface](#user-interface)
- [Deployment Strategy](#deployment-strategy)
- [Future Improvements](#future-improvements)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Ethical Considerations](#ethical-considerations)
- [Connect with Me](#connect-with-me)

---

## 📌 Project Overview

This project focuses on building an AI-powered system to verify employee identity through facial recognition. The system uses **face embeddings** and a deep learning model to classify whether a given input face matches a known identity (in this case, Arnold Schwarzenegger vs others).

It is designed for use cases in:

- Corporate access control
- Fintech and KYC verification
- Ed-tech and remote learning systems
- Any digital platform requiring face-based verification

---

## 🎥 Demo

<img src="demo/demo_ui.gif" width="70%" />

> **Note:** Full working prototype built using **Streamlit** is included.

---

## ❓ Problem Statement

Manual identity verification is slow, error-prone, and difficult to scale. With increasing cases of identity theft, there is a need for a secure, real-time, and automated solution.

**Objective:** Build a system that verifies whether a person’s face matches their registered identity using deep learning and facial embeddings.

---

## 📊 Dataset

- **Name:** `lfw_arnie_nonarnie.csv`
- Derived from: [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/)
- Each row is a **128-dimensional face embedding**
- Labels:
  - `1` → Arnold Schwarzenegger
  - `0` → Others

No raw images are used — embeddings are precomputed via FaceNet/Dlib.

---

## 🧠 Model Architecture

Developed using **Keras** with the following layers:

- Dense(128) + ReLU + Dropout(0.4)
- Dense(64) + ReLU + Dropout(0.3)
- Output: Dense(1) + Sigmoid

**Optimizer:** Adam  
**Loss:** Binary Crossentropy  
**Epochs:** 50 (with early stopping)  
**Batch Size:** 32

---

## 📈 Results

| Metric        | Value       |
|---------------|-------------|
| Test Accuracy | 78.95%      |
| Precision     | Acceptable  |
| Recall        | Acceptable  |
| Confusion Matrix | Good class separation |

> The system performs well for a binary classification task using facial embeddings.

---

## 💻 User Interface

Built a **Streamlit** prototype for real-time verification.

**Features:**

- Upload image of a face
- Predict if it's Arnold Schwarzenegger
- Show confidence score
- Easy to extend for enterprise access control

---

## ☁️ Deployment Strategy

- **Model Hosting:** Heroku / Render / AWS EC2
- **App Framework:** Streamlit or Flask
- **Database:** Encrypted PostgreSQL or SQLite
- **Security:** HTTPS, authentication, data encryption

---

## 🔮 Future Improvements

- Add webcam support using OpenCV
- Implement liveness detection (e.g., blink detection)
- Extend to multi-user classification
- Use advanced pretrained models (FaceNet, MobileNet)
- Edge deployment on Raspberry Pi or mobile

---

## 🛠️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/A289shek2004/face-id-employee-verification.git
cd face-id-employee-verification
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Launch the Streamlit app
bash
Copy
Edit
streamlit run app.py
**
⚙️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

TensorFlow / Keras

Streamlit (UI)

Matplotlib / Seaborn (for EDA)

StandardScaler, LabelEncoder

Confusion Matrix, Accuracy Plotting

🛡 Ethical Considerations
✅ Compliant with GDPR and India’s DPDP Act
✅ Explicit user consent before data usage
✅ Data encryption and secure storage
✅ No surveillance or non-consensual use

🙋‍♂️ Connect with Me
📧 Email: 1289shek2004@gmail.com

🔗 LinkedIn: Abhishek Gupta

💻 GitHub: A289shek2004

📂 Project Structure
bash
Copy
Edit
├── lfw_arnie_nonarnie.csv         # Dataset
├── reorganized_face_identifying.ipynb  # Model training & EDA
├── app.py                         # Streamlit UI prototype
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── demo/                         # (Optional) GIFs/screenshots

📢 License
This project is for educational purposes only and is not intended for commercial deployment without necessary modifications and approvals.**
