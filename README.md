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
pip install pandas numpy matplotlib seaborn tensorflow nltk scikit-learn emoji kaggle
```

You may also want to install `jupyterlab` if you use notebooks.

---

## Dataset (Kaggle)

This README assumes you are using the "Kaggle NLP with Disaster Tweets" dataset and that you will save the CSV files to the following paths in a Colab or local environment:

```py
train = pd.read_csv('/content/kaggle nlp/train.csv')
test = pd.read_csv('/content/kaggle nlp/test.csv')
```

### Downloading from Kaggle (CLI)

1. Install and configure Kaggle API (place `kaggle.json` in `~/.kaggle/`):

```bash
pip install kaggle
mkdir -p ~/.kaggle
# put kaggle.json here and set permissions
chmod 600 ~/.kaggle/kaggle.json
```

2. Download dataset (example):

```bash
kaggle competitions download -c nlp-getting-started -p /content/kaggle\ nlp
unzip '/content/kaggle nlp/nlp-getting-started.zip' -d '/content/kaggle nlp'
```

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
nltk.download('punkt')
nltk_packages = ['stopwords', 'wordnet', 'omw-1.4']
for p in nnltk_packages:
    try:
        nltk.download(p)
    except:
        pass

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
wnl = WordNetLemmatizer()

url_pattern = re.compile(r'https?://\S+|www\.\S+')
mention_pattern = re.compile(r'@\w+')
emoji_pattern = emoji.get_emoji_regexp()

def clean_tweet(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = text.lower()
    text = url_pattern.sub('', text)
    text = mention_pattern.sub('', text)
    # remove hashtags marker but keep the word
    text = re.sub(r'#', '', text)
    # remove emojis
    text = emoji_pattern.sub(r'', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    # lemmatize then stem (optional, choose one)
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)
```

Apply cleaning:

```python
train['clean_text'] = train['text'].apply(clean_tweet)
# If test has no labels, still clean for inference / validation
if 'text' in test.columns:
    test['clean_text'] = test['text'].apply(clean_tweet)
```

---

## Feature extraction: TF-IDF

Convert cleaned text to TF-IDF vectors. Use `TfidfVectorizer` from scikit-learn and limit features for speed.

```python
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)
X = vectorizer.fit_transform(train['clean_text'])
y = train['target']
```

---

## Train / validation split

Split data for training and validation. Set `random_state` for reproducibility.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Train Logistic Regression

Train a Logistic Regression classifier. Use `max_iter` large enough to converge and `class_weight='balanced'` if the dataset is imbalanced.

```python
clf = LogisticRegression(solver='saga', max_iter=5000, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)
print('Validation accuracy:', acc)

print('\nClassification report:')
print(classification_report(y_val, preds, digits=4))

print('\nConfusion matrix:')
print(confusion_matrix(y_val, preds))
```

### Reaching ~81.5% accuracy

Model performance depends on preprocessing, feature choices, and hyperparameters. With the steps above (TF-IDF with bigrams, `max_features=20000`, careful cleaning, and logistic regression with `saga` + `class_weight='balanced'`), many users obtain validation accuracies in the **~78–83%** range on this competition-style dataset. If you want to *target* `81.5%` specifically, try the following practical tips:

* Tune `max_features` (5k, 10k, 20k) and `ngram_range` ((1,1), (1,2)).
* Try `min_df=2` or `min_df=3`.
* Use `C` (inverse regularization) grid search for logistic regression (e.g., [0.01, 0.1, 1, 10]).
* Use simple feature engineering: length-of-text, num_hashtags, num_mentions, presence_of_url.
* Use stratified splits and cross-validation to avoid lucky splits.
* Use clean_text variations (only lemmatize, or only stem) and compare.

If you follow the recommended settings and tune `C` with 5-fold CV, you can typically reach or surpass **81.5%** on validation — but results vary by random seed and preprocessing.

---

## Example: printing a classification matrix / report

Below is an example (format) you will see after running the code. **This is an example output block — run the script to compute real values on your split.**

```
Validation accuracy: 0.8150

<img width="510" height="393" alt="image" src="https://github.com/user-attachments/assets/e9075db7-e1ef-4616-97ba-bc7cfd4d164d" />

Confusion matrix:
[[1680  320]
 [ 315 1185]]
```

> Note: numbers above are illustrative and show the format. Your actual confusion matrix and counts will depend on your split.

---

## Notebook / script structure

* `data/` : place `train.csv` and `test.csv` here (or point code to `/content/kaggle nlp/`)
* `notebooks/` : Jupyter notebook with EDA and model training
* `src/` : helper modules (preprocessing, features, modeling)
* `README.md` : this file

## Reproducibility tips

* Use `random_state=42` for splits and classifiers.
* Persist the fitted `TfidfVectorizer` using `joblib.dump(vectorizer, 'vectorizer.joblib')` and load it for inference.

## Troubleshooting

* If `nltk` downloads fail, run the downloader once in your environment:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

* If memory is tight, reduce `max_features` or use `HashingVectorizer` for streaming.

## License

This project is MIT licensed — include your preferred license.

---

If you want, I can also:

* add a ready-to-run `train.py` script (single-file) that implements everything above, or
* create a Jupyter notebook with the full code and plots.
