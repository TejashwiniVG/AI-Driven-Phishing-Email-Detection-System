import os
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler   # NEW
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)
from model_utils import text_preprocess

# Define labels
LABELS = ['legit', 'spam', 'phishing']
label_to_id = {l: i for i, l in enumerate(LABELS)}

def load_dataset():
    """
    Load dataset from CSV.
    Expects: data/email_dataset.csv with columns: subject, body, label
    """
    csv_path = 'data/email_dataset.csv'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No dataset found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure required columns
    required_cols = ['subject', 'body', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column.")

    # Fill missing values and reset index
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df = df.dropna(subset=['label']).reset_index(drop=True)

    # Optional: shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Dataset loaded successfully.")
    print("Class distribution:\n", df['label'].value_counts())
    return df

def main():
    df = load_dataset()
    
    # Combine subject + body and preprocess
    df['text'] = df['subject'] + ' ' + df['body']
    df['text_clean'] = df['text'].apply(text_preprocess)

    # Keep only valid labels
    df['label'] = df['label'].str.lower().str.strip()
    df = df[df['label'].isin(LABELS)].copy()
    df['label_id'] = df['label'].map(label_to_id)

    X = df['text_clean']
    y = df['label_id']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Before Oversampling:", y_train.value_counts())
    # Oversample minority classes   # NEW
    X_train_df = X_train.to_frame()
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train_df, y_train)
    X_train = X_resampled['text_clean']
    y_train = y_resampled
    print("After Oversampling:", y_train.value_counts())

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=40000)),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            multi_class='ovr'
        ))
    ])

    print("Training multi-class model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    # --- Extra Performance Metrics ---
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    kappa = cohen_kappa_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    try:
        roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
    except Exception:
        roc_auc = "N/A"

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Matthews Corrcoef (MCC): {mcc:.4f}")
    print(f"ROC-AUC (macro avg): {roc_auc}")

    print("\nClassification Report:\n", classification_report(y_test, preds, target_names=LABELS))
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)

    # --- Confusion Matrix Heatmap ---
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    print("\nSample prediction probabilities (first 5 test emails):")
    for i, prob in enumerate(probs[:5]):
        print(f"Email {i+1} -> {dict(zip(LABELS, prob))}")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump({'model': pipeline, 'labels': LABELS}, 'models/tfidf_lr_multiclass.joblib')
    print("Saved model to models/tfidf_lr_multiclass.joblib")

if __name__ == '__main__':
    main()
