# # predict_email_fixed.py
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import joblib

# # -------------------------------
# # Paths
# # -------------------------------
# MODEL_DIR = "models"
# MAX_LEN = 200  # must match training

# CNN_FILE = os.path.join(MODEL_DIR, "cnn_text_email.h5")
# RESNET_FILE = os.path.join(MODEL_DIR, "resnet_model.pth")
# LR_FILE = os.path.join(MODEL_DIR, "tfidf_lr_multiclass.joblib")
# TOKENIZER_FILE = os.path.join(MODEL_DIR, "cnn_tokenizer.joblib")
# LABELENC_FILE = os.path.join(MODEL_DIR, "label_encoder_resnet.joblib")

# # -------------------------------
# # Define ResNet1D class (must match training)
# # -------------------------------
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=1)
#         self.bn2 = nn.BatchNorm1d(out_channels)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return F.relu(out)

# class TextResNet(nn.Module):
#     def __init__(self, vocab_size, embed_dim, num_classes):
#         super(TextResNet, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#         )
#         self.resblock = nn.Sequential(
#             nn.Conv1d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm1d(128),
#         )
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.embedding(x).permute(0, 2, 1)
#         out = self.layer1(x)
#         residual = out
#         out = self.resblock(out)
#         out += residual
#         out = torch.relu(out)
#         out = torch.max(out, dim=2)[0]  # Global Max Pool
#         out = self.fc(out)
#         return out


# # -------------------------------
# vocab = joblib.load("models/vocab_resnet.joblib")
# label_encoder = joblib.load("models/label_encoder_resnet.joblib")
# # CNN + Tokenizer
# cnn_model = load_model(CNN_FILE) if os.path.isfile(CNN_FILE) else None
# tokenizer = joblib.load(TOKENIZER_FILE) if os.path.isfile(TOKENIZER_FILE) else None



# # Logistic Regression + TF-IDF
# lr_model, tfidf_vectorizer = None, None
# if os.path.isfile(LR_FILE):
#     loaded = joblib.load(LR_FILE)
#     if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
#         lr_model, tfidf_vectorizer = loaded
#     elif isinstance(loaded, dict):
#         lr_model = loaded.get("model") or loaded.get("lr") or loaded.get("clf")
#         tfidf_vectorizer = loaded.get("vectorizer") or loaded.get("tfidf")
#     else:
#         raise ValueError("LR file format not recognized. Must contain model and TF-IDF vectorizer.")

# # ResNet PyTorch
# # ResNet PyTorch (TextResNet)
# resnet_model = None
# if os.path.isfile(RESNET_FILE):
#     print(f"Loading ResNet from {RESNET_FILE}")
#     resnet_model = TextResNet(len(vocab), 128, len(label_encoder.classes_))
#     state_dict = torch.load(RESNET_FILE, map_location="cpu")
#     resnet_model.load_state_dict(state_dict)
#     resnet_model.eval()
# else:
#     print("[WARN] ResNet model not found.")


# # -------------------------------
# # Prediction function
# # -------------------------------
# def predict_email(text):
#     results = []

#     # Prepare input for CNN/ResNet
#     if tokenizer is not None and (cnn_model or resnet_model):
#         seq = tokenizer.texts_to_sequences([text])
#         padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

#     # CNN Prediction
#     if cnn_model is not None:
#         probs = cnn_model.predict(padded, verbose=0)[0]
#         idx = int(np.argmax(probs))
#         conf = float(np.max(probs))
#         label = label_encoder.inverse_transform([idx])[0] if label_encoder else idx
#         results.append({"model":"CNN", "label_idx":idx, "label":label, "confidence":conf})

#     # ResNet Prediction
#     if resnet_model is not None:
#         x_resnet = torch.tensor(padded, dtype=torch.long)
#         with torch.no_grad():
#             logits = resnet_model(x_resnet)
#             probs = torch.softmax(logits, dim=1).numpy()[0]
#         idx = int(np.argmax(probs))
#         conf = float(np.max(probs))
#         label = label_encoder.inverse_transform([idx])[0] if label_encoder else idx
#         results.append({"model":"ResNet", "label_idx":idx, "label":label, "confidence":conf})

#     # Logistic Regression Prediction
#     if lr_model is not None and tfidf_vectorizer is not None:
#         x_tfidf = tfidf_vectorizer.transform([text])
#         probs = lr_model.predict_proba(x_tfidf)[0]
#         idx = int(np.argmax(probs))
#         conf = float(np.max(probs))
#         label = label_encoder.inverse_transform([idx])[0] if label_encoder else idx
#         results.append({"model":"LogisticRegression", "label_idx":idx, "label":label, "confidence":conf})

#     # Select most confident
#     best = max(results, key=lambda r: r["confidence"])
#     return {"best_label": best["label"],
#             "best_model": best["model"],
#             "best_confidence": best["confidence"],
#             "per_model": results}

# # -------------------------------
# # CLI Test
# # -------------------------------
# if __name__ == "__main__":
#     sample_email = "Subject: Your account has been suspended. Body: Please click the link to restore access."
#     out = predict_email(sample_email)
#     print("Best prediction:", out["best_label"])
#     print("Model used:", out["best_model"])
#     print("Confidence:", f"{out['best_confidence']:.4f}")
#     print("\nPer-model details:")
#     for r in out["per_model"]:
#         print(f" - {r['model']}: {r['label']} (conf={r['confidence']:.4f})")
# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib, os, time
from datetime import datetime
from collections import Counter
from model_utils import text_preprocess
from security_checks import compute_authenticity, analyze_urls_in_text

app = Flask(__name__)
CORS(app)

# -------------------------------
# Paths
# -------------------------------
MODEL_DIR = "models"
MAX_LEN = 200

CNN_FILE = os.path.join(MODEL_DIR, "cnn_text_email.h5")
RESNET_FILE = os.path.join(MODEL_DIR, "resnet_model.pth")
LR_FILE = os.path.join(MODEL_DIR, "tfidf_lr_multiclass.joblib")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "cnn_tokenizer.joblib")
VOCAB_FILE = os.path.join(MODEL_DIR, "vocab_resnet.joblib")
LABELENC_FILE = os.path.join(MODEL_DIR, "label_encoder_resnet.joblib")

history = []

# -------------------------------
# Load models and tokenizers
# -------------------------------
# Logistic Regression + TF-IDF
lr_model, tfidf_vectorizer = None, None
if os.path.isfile(LR_FILE):
    loaded = joblib.load(LR_FILE)
    if isinstance(loaded, dict):
        lr_model = loaded.get("model")
        tfidf_vectorizer = loaded.get("vectorizer")

# CNN
cnn_model = load_model(CNN_FILE) if os.path.isfile(CNN_FILE) else None
tokenizer = joblib.load(TOKENIZER_FILE) if os.path.isfile(TOKENIZER_FILE) else None

# ResNet
class TextResNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.layer1 = nn.Sequential(
            nn.Conv1d(embed_dim, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.resblock = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        out = self.layer1(x)
        residual = out
        out = self.resblock(out)
        out += residual
        out = torch.relu(out)
        out = torch.max(out, dim=2)[0]
        out = self.fc(out)
        return out

vocab = joblib.load(VOCAB_FILE)
label_encoder = joblib.load(LABELENC_FILE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = None
if os.path.isfile(RESNET_FILE):
    resnet_model = TextResNet(len(vocab), 128, len(label_encoder.classes_)).to(device)
    resnet_model.load_state_dict(torch.load(RESNET_FILE, map_location=device))
    resnet_model.eval()

# -------------------------------
# Flask routes
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    subject = data.get('subject','')
    body = data.get('body','')
    headers_text = data.get('headers','')
    text_raw = (subject + " " + body).strip()
    text = text_preprocess(text_raw)

    results = []

    # ---------------- CNN Prediction ----------------
    if cnn_model is not None and tokenizer is not None:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        probs = cnn_model.predict(padded, verbose=0)[0]
        idx = int(probs.argmax())
        label = label_encoder.inverse_transform([idx])[0]
        conf = float(np.max(probs))
        results.append({"model":"CNN","label":label,"confidence":conf})

    # ---------------- ResNet Prediction ----------------
    if resnet_model is not None:
        x_resnet = torch.tensor(padded, dtype=torch.long).to(device)
        with torch.no_grad():
            logits = resnet_model(x_resnet)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        label = label_encoder.inverse_transform([idx])[0]
        conf = float(np.max(probs))
        results.append({"model":"ResNet","label":label,"confidence":conf})

    # ---------------- Logistic Regression Prediction ----------------
    if lr_model is not None and tfidf_vectorizer is not None:
        x_tfidf = tfidf_vectorizer.transform([text])
        probs = lr_model.predict_proba(x_tfidf)[0]
        idx = int(probs.argmax())
        label = label_encoder.inverse_transform([idx])[0]
        conf = float(np.max(probs))
        results.append({"model":"LogisticRegression","label":label,"confidence":conf})

    # ---------------- Majority Vote ----------------
    labels = [r["label"] for r in results]
    counter = Counter(labels)
    majority_label, majority_count = counter.most_common(1)[0]
    tied_labels = [label for label,count in counter.items() if count==majority_count]
    if len(tied_labels)>1:
        # tie-breaker: highest confidence
        filtered = [r for r in results if r["label"] in tied_labels]
        best = max(filtered, key=lambda r: r["confidence"])
    else:
        best = max([r for r in results if r["label"]==majority_label], key=lambda r: r["confidence"])

    final_label = best["label"]
    best_model = best["model"]
    best_conf = best["confidence"]

    # ---------------- Security & URL analysis ----------------
    auth = compute_authenticity(headers_text)
    urls = analyze_urls_in_text(text_raw, claimed_domain=auth.get('from_domain',''))

    # Phishing score (optional)
    phishing_score = (
        0.6*(results[0]['confidence'] if results else 0) +
        0.25*auth.get('auth_risk',0)/100.0 +
        0.15*urls.get('url_risk',0)/100.0
    )
    phishing_score = max(0, min(1, phishing_score))

    # ---------------- Save history ----------------
    item = {
        'id': int(time.time()*1000),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subject': subject[:120],
        'final_hint': final_label,
        'best_model': best_model,
        'best_confidence': float(best_conf),
        'per_model': results,
        'auth': auth,
        'urls': urls,
        'phishing_score': float(phishing_score)
    }
    history.append(item)
    history[:] = history[-100:]

    return jsonify(item)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'items': history[::-1]})

# -------------------------------

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
