from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import tldextract
from flask_cors import CORS
import joblib, os, time
from datetime import datetime
from model_utils import text_preprocess
from lime.lime_text import LimeTextExplainer
from security_checks import compute_authenticity, analyze_urls_in_text
from train_resnet import TextResNet, encode
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('models', 'tfidf_lr_multiclass.joblib')
ARTIFACT = None
LABELS = []
explainer = None
history = []

def load_model():
    global ARTIFACT, LABELS, explainer
    if os.path.exists(MODEL_PATH):
        ARTIFACT = joblib.load(MODEL_PATH)
        LABELS = ARTIFACT.get('labels', ['legit', 'spam', 'phishing'])
        explainer = LimeTextExplainer(class_names=LABELS)
        print("Model loaded.")
    else:
        print("WARNING: model not found. Train with train_multi.py")

load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if ARTIFACT is None:
        return jsonify({'error': 'Model not loaded. Run train_multi.py first.'}), 500

    data = request.get_json(force=True)
    subject = data.get('subject', '') or ''
    body = data.get('body', '') or ''
    headers_text = data.get('headers', '') or ''
    text_raw = (subject + ' ' + body).strip()
    text = text_preprocess(text_raw)
    model = ARTIFACT['model']

    # Predict probabilities
    probs = model.predict_proba([text])[0].tolist()
    pred_id = int(max(range(len(probs)), key=lambda i: probs[i]))
    label = LABELS[pred_id]
    per_class = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    
    # Compute security features
    auth = compute_authenticity(headers_text)
    urls = analyze_urls_in_text(text_raw, claimed_domain=auth.get('from_domain', ''))

    # Compute phishing score
    phishing_score = (
        0.6* per_class.get('phishing', 0.0)
        + 0.25 * auth['auth_risk'] / 100.0
        + 0.15 * urls['url_risk'] / 100.0
    )
    phishing_score = max(0.0, min(1.0, phishing_score))

    # Determine final label
    final_hint = label
    SPAM_DOMAINS = ["angelbroking.in", "someother-spam.com"]
    from_domain = auth.get('from_domain', '').lower()

    if from_domain in SPAM_DOMAINS:
        final_hint = 'spam'
    elif per_class.get('spam', 0.0) >= 0.5:
        final_hint = 'spam'
    elif phishing_score >= 0.6 and per_class.get('legit', 0.0) < 0.5:
        final_hint = 'phishing'

    # Prepare result
    item = {
        'id': int(time.time() * 1000),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'subject': subject[:120],
        'label': label,
        'probs': per_class,
        'auth': auth,
        'urls': urls,
        'phishing_score': float(phishing_score),
        'final_hint': final_hint
    }

    # Keep history
    history.append(item)
    history[:] = history[-100:]

    return jsonify(item)
@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    data = request.json
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Basic heuristics for demonstration
    risk = 0
    urls = []

    # Example scoring based on suspicious keywords & URL length
    suspicious_keywords = [
    "secure", "update", "login", "verify", "account", "paypal", "bank", 
    "confirm", "password", "signin", "security", "ebay", "amazon", 
    "invoice", "payment", "urgent", "alert", "click", "reset", 
    "unauthorized", "suspended", "activation", "credit", "debit", 
    "support", "service", "notification", "customer", "bill", 
    "verify-now", "confirm-account", "login-alert", "account-update"
]   
    url_lower = url.lower()
    url_domain = tldextract.extract(url).domain
    url_length = len(url)
    entropy = round(len(set(url)) / max(1, len(url)), 2)
    matches = sum(1 for kw in suspicious_keywords if kw in url_lower)

    keyword_score = min(1.0, matches / 5)
    domain_score = 1 if any(b in url_domain.lower() for b in brand_keywords) else 0

    # Risk is combination of length, keywords, entropy (normalized 0-1)
    risk = 0.4 * keyword_score + 0.35 * domain_score + 0.15 * entropy + 0.1 * min(url_length/50, 1)
    risk = min(max(risk, 0), 1)


    print("Calculated risk:", risk)

    urls.append({
        "url": url,
        "risk": round(risk, 2)
    })

    return jsonify({
        "status": "success",
        "url_risk": round(risk, 2),
        "urls": urls
    })
    
@app.route('/explain', methods=['POST'])
def explain():
    if ARTIFACT is None:
        return jsonify({'error': 'Model not loaded. Run train_multi.py first.'}), 500

    data = request.get_json(force=True)
    subject = data.get('subject', '') or ''
    body = data.get('body', '') or ''
    text_raw = (subject + ' ' + body).strip()
    text = text_preprocess(text_raw)
    model = ARTIFACT['model']

    # Get predicted class index
    probs = model.predict_proba([text])[0].tolist()
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

    # Explain using LIME
    exp = LimeTextExplainer(class_names=LABELS).explain_instance(
        text, model.predict_proba, num_features=10, labels=[pred_idx]
    )
    weights = exp.as_list(label=pred_idx)

    return jsonify({
        'tokens': [{'token': token, 'weight': float(weight)} for (token, weight) in weights],
        'predicted_label': LABELS[pred_idx]
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({'items': history[::-1]})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = joblib.load("models/vocab_resnet.joblib")
label_encoder = joblib.load("models/label_encoder_resnet.joblib")

resnet_model = TextResNet(len(vocab), 128, len(label_encoder.classes_)).to(device)
resnet_model.load_state_dict(torch.load("models/resnet_model.pth", map_location=device))
resnet_model.eval()

# ---------------- CNN Model ----------------
MAX_LEN = 200
cnn_model = tf.keras.models.load_model("models/cnn_text_phishing.h5")
cnn_tokenizer = joblib.load("models/cnn_tokenizer.joblib")


@app.route('/predict_resnet', methods=['POST'])
def predict_resnet():
    data = request.get_json(force=True)
    subject = data.get('subject', '')
    body = data.get('body', '')
    text = (subject + " " + body).strip()
    encoded = torch.tensor([encode(text)], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = resnet_model(encoded)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    pred_id = int(probs.argmax())
    label = label_encoder.classes_[pred_id]
    per_class = {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))}

    return jsonify({"prediction": label, "probabilities": per_class})

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    data = request.get_json(force=True)
    subject = data.get('subject', '')
    body = data.get('body', '')
    text = (subject + " " + body).strip()

    # Convert text → sequence → padded
    seq = cnn_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Predict
    with tf.device("/CPU:0"):  # avoids GPU conflicts with PyTorch
        probs = cnn_model.predict(padded)[0]

    # Handle multi-class
    if len(probs) > 1:
        pred_id = int(probs.argmax())
        label = LABELS[pred_id] if LABELS else str(pred_id)
        per_class = {LABELS[i]: float(probs[i]) for i in range(len(probs))}
    else:  # binary phishing vs not
        pred_id = 1 if probs[0] >= 0.5 else 0
        label = "phishing" if pred_id == 1 else "legit"
        per_class = {"legit": 1 - float(probs[0]), "phishing": float(probs[0])}

    return jsonify({"prediction": label, "probabilities": per_class})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
