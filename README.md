# Disaster-Tweet

# Disaster Tweets Classification

This repository trains a **Logistic Regression** model to detect whether a tweet describes a real disaster or not (binary classification). The README includes all steps: environment setup, data download (Kaggle), preprocessing with NLP, TF-IDF vectorization, model training, evaluation (classification matrix / report) and reproducibility notes.

---

## Dependencies

The project uses the following Python libraries (your imports):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import re
import string
import emoji

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

> Recommended Python version: **3.8+**

Install requirements (example):

```bash
!pip install emoji
!pip install nltk```

---

## Dataset (Kaggle)

This README assumes you are using the "Kaggle NLP with Disaster Tweets" dataset and that you will save the CSV files to the following paths in a Colab or local environment:


### Downloading from Kaggle (CLI)

1. Install and configure Kaggle API (place `kaggle.json` in `~/.kaggle/`):

```bash
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

2. Download dataset (example):

```bash
!kaggle datasets download -d dipankarmitra/natural-language-processing-with-disaster-tweets
!unzip datasets/dipankarmitra/natural-language-processing-with-disaster-tweets```

Adjust paths if you use a different dataset source.

---

## Preprocessing pipeline

Below is a compact, reproducible preprocessing function that:

* lowers text
* removes URLs, mentions, hashtags (keeps words), emojis
* removes punctuation
* tokenizes, removes stopwords
* lemmatizes (WordNet) and stems (Porter) optionally

```python
import nltk
nltk_packages = ['stopwords', 'wordnet', 'omw-1.4']

def clean_tweet(text):

    # Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 3. Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # 4. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 5. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 6. Tokenize
    words = text.split()

    # 7. Remove stopwords
    words = [word for word in words if word not in stop_words]

    # 8A. Stemming
    stemmed_words = [stemmer.stem(word) for word in words]

    # 8B. Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(lemmatized_words)
```

Apply cleaning:

```python
train['clean_text'] = train['text'].apply(clean_tweet)

```

---

## Feature extraction: TF-IDF

Convert cleaned text to TF-IDF vectors. Use `TfidfVectorizer` from scikit-learn and limit features for speed.

```python
vectorizer = TfidfVectorizer(max_features=40000)
X = vectorizer.fit_transform(train['clean_text'])
y = train['target']
```

---

## Train / validation split

Split data for training and validation. Set `random_state` for reproducibility.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Train Logistic Regression

Train a Logistic Regression classifier. Use `max_iter` large enough to converge and `class_weight='balanced'` if the dataset is imbalanced.

```python
clf = LogisticRegression(max_iter=300)
clf.fit(X_train, y_train)

y_test = clf.predict(X_val)
acc = accuracy_score(y_test, y_pred)
print('Validation accuracy:', acc)

print('\nConfusion matrix:')
print(confusion_matrix(y_test, y_pred))
```

### Reaching ~81.5% accuracy

---

## Example: printing a classification matrix / report

Below is an example (format) you will see after running the code. **This is an example output block — run the script to compute real values on your split.**

```
Validation accuracy: 0.8115561391989494

Confusion matrix:
[[789  89]
 [ 198 447]]
```

> Note: numbers above are illustrative and show the format. Your actual confusion matrix and counts will depend on your split.

---


## Reproducibility tips

* Use `random_state=42` for splits and classifiers.

## Troubleshooting

* If `nltk` downloads fail, run the downloader once in your environment:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```
