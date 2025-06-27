from utils.preprocess import preprocess_texts
from models.model import build_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
df = pd.read_csv("tweet.csv")

# Clean and preprocess data
texts, labels = preprocess_texts(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Pad sequences
X_train_pad = pad_sequences(X_train, maxlen=40, padding='post')
X_test_pad = pad_sequences(X_test, maxlen=40, padding='post')

# Build and train model
model = build_model()
model.fit(X_train_pad, y_train, epochs=5, validation_split=0.1)

# Evaluate
y_pred = model.predict(X_test_pad)
y_pred = (y_pred > 0.5).astype(int).flatten()
print("Test Accuracy:", accuracy_score(y_test, y_pred))
