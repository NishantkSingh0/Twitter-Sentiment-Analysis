# 🐦 Tweet Sentiment Analysis

Analyze and classify sentiments in tweets using NLP, Transformer-based embeddings, clustering, and deep learning.

<br><br>

## 📌 Project Overview

This project aims to perform **sentiment classification** (Happy 😄 / Angry 😠) on tweet data using a blend of **traditional ML** and **deep learning techniques**. The focus is on **clean data preparation**, **semantic embeddings using Transformers**, and building a **high-accuracy model** even with a small dataset.

<br><br>

## 🧠 Workflow Summary

<br>

1. **Dataset Source**
   Used a Kaggle [tweet sentiment dataset](https://www.kaggle.com/datasets/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv)

<br>

2. **Data Preprocessing**

   * Removed irrelevant features: `ID`, `Date`, `Query?`, and `Names`
   * Applied regular expression (regex) techniques for text cleaning
   * Limited all sequences to **40 tokens**; rows with longer text were dropped
  ![image](https://github.com/user-attachments/assets/c707cadf-8041-4bc9-977f-d1c799e5a7aa)


<br>

3. **Feature Representation**

   * Used pretrained Transformer model: `all-MiniLM-L6-v2` from [SentenceTransformers](https://www.sbert.net/)
   * Converted cleaned tweets into **semantic vector embeddings**

<br>

4. **Clustering (Unsupervised Analysis)**

   * Applied **K-Means** and **DBSCAN** clustering to explore semantic groupings in sentiment space

<br>

5. **Deep Learning Classification**
 ```
    model=tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.LayerNormalization(epsilon=1e-6),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
```

   * Built a custom `tf.keras.Sequential` model with:
     * Normalization and Dropout layers
     * Optimized parameter tuning

   * Trained on **5000 samples** (2500 Angry, 2500 Happy)
   * Achieved **97.56% accuracy**

<br><br>

## 🛠️ Technologies & Libraries

<div align="center"> <img src="https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" /> <img src="https://img.shields.io/badge/TensorFlow-FE6F00?style=for-the-badge&logo=tensorflow&logoColor=white" /> <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" /> <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" /> <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/SentenceTransformers-00599C?style=for-the-badge&logo=sentence-transformers&logoColor=white" /> <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black" /> <img src="https://img.shields.io/badge/Seaborn-5389A6?style=for-the-badge&logo=seaborn&logoColor=white" /> </div>

<br><br>

## 📊 Model Performance

* **Accuracy**: `97.56%`
* Demonstrates strong model generalization even on a relatively small and balanced dataset.
 
<br><br>

## 📁 File Structure

```
📦tweet-sentiment-analysis/   
 ┣ 📜TweeterSentimentAnalysis.ipynb     # Jupyter notebook for exploration & prototyping   
 ┣ 📂Tweet_sentiment_analysis/          # Python module for production code   
 ┃ ┣ 📂models/   
 ┃ ┃ ┣ 📜__init__.py    
 ┃ ┃ ┗ 📜model.py                       # Contains model-building function   
 ┃ ┣ 📂utils/    
 ┃ ┃ ┣ 📜__init__.py   
 ┃ ┃ ┗ 📜preprocess.py                  # Text cleaning, tokenization   
 ┃ ┣ 📜main.py                          # Main script to train & evaluate the model   
 ┃ ┗ 📜requirements.txt                 # Dependencies used in module   
 ┗ 📜README.md                          # Project overview, usage instructions   
```

<br><br>

## 🔍 Example Embedding & Clustering Visualization

![image](https://github.com/user-attachments/assets/33dab561-ac91-448b-954e-becb14eabcef)
![image](https://github.com/user-attachments/assets/60d876d8-591d-4f4e-98ea-ba113faf25a0)

<br><br>

## 🚀 How to Run

```bash
git clone https://github.com/nishantksingh0/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
pip install -r requirements.txt
cd Tweet_sentiment_analysis
python main.py
```

<br>

Live Colab link: <a href="https://colab.research.google.com/drive/1O56ZFA6zkZfhfUnjqdHeSStWm_XpyLg8?usp=sharing" target="_blank">TweeterSentimentAnalysis<a>


<br><br>

## ✅ Results & Insights

* Pretrained embeddings significantly improved semantic understanding
* Careful data preprocessing ensured high performance on even a small dataset
* Potential to expand for multi-class sentiment classification (e.g., sad, excited, neutral)

<br><br>

## 📌 Future Work

* Use full dataset to further test generalizability
* Deploy the model as a web app (e.g., using Flask or Streamlit)
* Try zero-shot sentiment classification with LLMs

<br><br>

## 📬 Contact

For questions or collaborations, reach out via [LinkedIn](https://www.linkedin.com/nishantksingh1) or [email](mailto:nishantsingh.talk@gmail.com).

