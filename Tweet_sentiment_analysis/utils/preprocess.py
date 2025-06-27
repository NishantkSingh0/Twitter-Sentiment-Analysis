import re
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def preprocess_texts(df):
    df = df.drop(columns=['id', 'date', 'flag', 'user'], errors='ignore')
    df['clean_text'] = df['text'].apply(clean_text)

    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    
    # Filter out sequences longer than 40
    filtered_sequences = []
    filtered_labels = []
    for seq, label in zip(sequences, df['target']):
        if len(seq) <= 40:
            filtered_sequences.append(seq)
            filtered_labels.append(label)

    return filtered_sequences, filtered_labels
