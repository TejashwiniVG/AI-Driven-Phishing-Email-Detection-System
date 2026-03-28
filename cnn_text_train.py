# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# import os
# import joblib   # For saving/loading tokenizer
# from tensorflow.keras.utils import to_categorical

# # -------------------------------
# # 1. Load dataset
# # -------------------------------
# # CSV must have: subject, body, label (phish, spam, legit)
# data = pd.read_csv("data/email_dataset.csv")

# # Combine subject + body as text
# texts = (data["subject"].astype(str) + " " + data["body"].astype(str)).values

# # Encode labels (phish/spam/legit -> 0/1/2)
# le = LabelEncoder()
# labels = le.fit_transform(data["label"].astype(str))  # phish=0, spam=1, legit=2
# y = to_categorical(labels)  # One-hot encoding for multi-class

# # -------------------------------
# # 2. Tokenization & Padding
# # -------------------------------
# max_words = 10000   # Vocabulary size
# max_len = 200       # Max words per email

# tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
# tokenizer.fit_on_texts(texts)

# sequences = tokenizer.texts_to_sequences(texts)
# X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# # -------------------------------
# # 3. Train-test split
# # -------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -------------------------------
# # 4. CNN Model for Text
# # -------------------------------
# model = Sequential([
#     Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
#     Conv1D(filters=128, kernel_size=5, activation='relu'),
#     GlobalMaxPooling1D(),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(3, activation='softmax')   # Multi-class output: phish, spam, legit
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # -------------------------------
# # 5. Train model
# # -------------------------------
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=5,
#     batch_size=32
# )

# # -------------------------------
# # 6. Save model + tokenizer
# # -------------------------------
# # Create "models" folder if it doesn’t exist
# os.makedirs("models", exist_ok=True)

# # Save trained model
# model.save("models/cnn_text_email.h5")
# print("✅ Model saved to models/cnn_text_email.h5")

# # Save tokenizer
# joblib.dump(tokenizer, "models/cnn_tokenizer.joblib")
# print("✅ Tokenizer saved to models/cnn_tokenizer.joblib")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import joblib   # For saving/loading tokenizer and label encoder

# -------------------------------
# 1. Load dataset
# -------------------------------
# CSV must have: subject, body, label (phish, spam, legit)
data = pd.read_csv("data/email_dataset.csv")

# Combine subject + body as text
texts = (data["subject"].astype(str) + " " + data["body"].astype(str)).values

# -------------------------------
# 2. Label Encoding (Global Consistency)
# -------------------------------
# Define consistent class labels globally
ALL_CLASSES = ["phish", "spam", "legit"]

# Fit label encoder on all possible labels (even if one is missing)
le = LabelEncoder()
le.fit(ALL_CLASSES)

# Transform labels
labels = le.transform(data["label"].astype(str))

# One-hot encode labels for multi-class classification
y = to_categorical(labels, num_classes=len(ALL_CLASSES))

# -------------------------------
# 3. Tokenization & Padding
# -------------------------------
max_words = 10000   # Vocabulary size
max_len = 200       # Max words per email

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# -------------------------------
# 4. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. CNN Model for Text
# -------------------------------
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(ALL_CLASSES), activation='softmax')   # Always 3 outputs: phish, spam, legit
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------------------
# 6. Train model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# -------------------------------
# 7. Save model, tokenizer, and label encoder
# -------------------------------
os.makedirs("models", exist_ok=True)

# Save trained model
model.save("models/cnn_text_email.h5")
print("✅ Model saved to models/cnn_text_email.h5")

# Save tokenizer
joblib.dump(tokenizer, "models/cnn_tokenizer.joblib")
print("✅ Tokenizer saved to models/cnn_tokenizer.joblib")

# Save label encoder
joblib.dump(le, "models/cnn_label_encoder.joblib")
print("✅ Label encoder saved to models/cnn_label_encoder.joblib")

# -------------------------------
# 8. Sanity check (optional)
# -------------------------------
print("\n✅ Training complete!")
print("Classes:", list(le.classes_))
print("Example label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
