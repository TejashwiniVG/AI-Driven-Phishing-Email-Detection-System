import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# -------------------
# Load dataset
# -------------------
df = pd.read_csv("data/email_dataset.csv")
df['combined_text'] = df['subject'].astype(str) + " " + df['body'].astype(str)
texts = df['combined_text'].tolist()
labels = df['label'].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Save label encoder
os.makedirs("models", exist_ok=True)
joblib.dump(label_encoder, "models/label_encoder_resnet.joblib")

# -------------------
# Tokenizer + Vocabulary
# -------------------
from collections import Counter
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

counter = Counter()
for t in texts:
    counter.update(tokenize(t))

# Build vocab (only most common words)
vocab = {word: idx+1 for idx, (word, _) in enumerate(counter.most_common(20000))}
vocab["<PAD>"] = 0

joblib.dump(vocab, "models/vocab_resnet.joblib")

def encode(text, max_len=200):
    tokens = tokenize(text)
    ids = [vocab.get(t, 0) for t in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

X = [encode(t) for t in texts]
y = labels

# -------------------
# Dataset + DataLoader
# -------------------
class EmailDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = EmailDataset(X_train, y_train)
val_ds = EmailDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# -------------------
# ResNet-like Text Model
# -------------------
class TextResNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextResNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.layer1 = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.resblock = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        out = self.layer1(x)
        residual = out
        out = self.resblock(out)
        out += residual
        out = torch.relu(out)
        out = torch.max(out, dim=2)[0]  # Global Max Pooling
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextResNet(len(vocab), 128, len(label_encoder.classes_)).to(device)

# -------------------
# Training
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss={total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/resnet_model.pth")
print("✅ ResNet model saved to models/resnet_model.pth")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import numpy as np

model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

# -------------------
# Confusion Matrix
# -------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# -------------------
# Classification Report
# -------------------
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# -------------------
# ROC Curve & AUC (for multi-class one-vs-rest)
# -------------------
y_true_bin = np.zeros((len(y_true), len(label_encoder.classes_)))
for i, label in enumerate(y_true):
    y_true_bin[i, label] = 1

plt.figure(figsize=(7, 6))
for i, class_name in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(y_probs)[:, i])
    auc = roc_auc_score(y_true_bin[:, i], np.array(y_probs)[:, i])
    plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend()
plt.show()

# -------------------
# Bar Chart for Metrics
# -------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}

plt.figure(figsize=(6, 5))
plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "orange", "green", "red"])
plt.title("Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()