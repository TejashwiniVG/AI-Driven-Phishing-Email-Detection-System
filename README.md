# AI Email Guard — Secure Edition

Adds **email authentication checks** (SPF/DKIM/DMARC) and **URL risk analysis** on top of the multi-class ML model.

## Features
- Multi-class ML: `legit`, `spam`, `phishing`
- LIME explanation (top tokens)
- **Authentication signals**: SPF presence, DKIM header presence, DMARC policy, From vs Reply-To mismatch
- **URL risk analysis**: suspicious TLDs, IP-in-URL, punycode, too many dots, domain mismatch
- Blended phishing score (ML + security signals)
- Real-time dashboard of recent analyses

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train_multi.py
```
Saves `models/tfidf_lr_multiclass.joblib`.

## Run
```bash
python app.py
```
Open http://127.0.0.1:5000

## Notes
- SPF/DMARC checks require DNS resolution; if you're offline, these will default to "not found".
- DKIM verification requires the raw email with headers and body; this app only checks presence of the `DKIM-Signature` header.
- For production, consider verifying DKIM with the raw MIME message and adding Safe Browsing/PhishTank lookups.
