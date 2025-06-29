import gradio as gr
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = load_model("Model.h5")

def classify_sentiment(text):
    embedding = embedder.encode(text, show_progress_bar=False)
    embedding = np.expand_dims(embedding, axis=0)  # (1, 384)
    pred = model.predict(embedding)[0][0]
    label = "Positive" if pred > 0.5 else "Negative"
    return f"Prediction: {label} (Score: {pred:.2f})"

interface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter a tweet..."),
    outputs=gr.Textbox(),
    title="Tweet Sentiment Classifier",
    description="Uses all-MiniLM-L6-v2 to convert your text into a meaningful vector and then classifies it as positive or negative sentiment using a trained deep Sequential model. \n👉 [View Source on GitHub](https://github.com/nishantksingh0/Twitter-Sentiment-Analysis)",
)

interface.launch()
