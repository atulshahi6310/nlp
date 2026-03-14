<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gensim-Word2Vec-blue?style=for-the-badge" />
</p>

<h1 align="center">📖 NLP Learning Booklet</h1>

<p align="center">
  <b>A comprehensive, hands-on guide to Natural Language Processing — from first principles to real-world projects.</b>
</p>

<p align="center">
  <i>Built with NLTK · Scikit-Learn · Gensim · Pandas · NumPy</i>
</p>

---

## 📑 Table of Contents

| #  | Chapter | Topics Covered |
|----|---------|---------------|
| 1  | [🔤 Tokenization](#chapter-1--tokenization) | Sentence, Word, Wordpunct, Treebank tokenizers |
| 2  | [🧹 Text Processing](#chapter-2--text-processing) | Stemming, Lemmatization, Stopwords, POS Tagging, NER |
| 3  | [🎒 Bag of Words (BoW)](#chapter-3--bag-of-words-bow) | CountVectorizer, Binary BoW, N-Grams, Spam Classification |
| 4  | [📊 TF-IDF](#chapter-4--tf-idf) | TfidfVectorizer, N-Gram TF-IDF, Feature weighting |
| 5  | [🧠 Word2Vec](#chapter-5--word2vec) | Word embeddings, Similarity, Vector arithmetic |

---

## Chapter 1 — 🔤 Tokenization

> **Tokenization** is the process of breaking raw text into smaller meaningful units called **tokens** — the very first step in any NLP pipeline.

📂 **Directory:** [`tokenization/`](tokenization/)

### 📘 What You'll Learn

#### 1.1 Sentence Tokenization (`sent_tokenize`)
- Splits a paragraph into **individual sentences**
- Uses the **Punkt Sentence Tokenizer** internally
- Detects sentence boundaries using `.` `!` `?` while avoiding false splits at abbreviations (Dr., Mr., etc.)

```python
from nltk.tokenize import sent_tokenize
sent_tokenize("Hello Atul. You are learning NLP! That's great.")
# → ['Hello Atul.', 'You are learning NLP!', "That's great."]
```

#### 1.2 Word Tokenization (`word_tokenize`)
- Splits text into **words and punctuation** tokens
- Linguistically smart — handles contractions properly (`"don't"` → `"do"`, `"n't"`)

```python
from nltk.tokenize import word_tokenize
word_tokenize("I don't like NLP!")
# → ['I', 'do', "n't", 'like', 'NLP', '!']
```

#### 1.3 Wordpunct Tokenization (`wordpunct_tokenize`)
- Regex-based, more aggressive splitting
- Splits all punctuation as separate tokens (`"don't"` → `"don"`, `"'"`, `"t"`)

#### 1.4 TreebankWordTokenizer
- Based on **Penn Treebank** linguistic rules
- Handles contractions like `word_tokenize` and keeps decimal numbers intact (`$5.50`)

### 🔥 Comparison Table

| Feature | `sent_tokenize` | `word_tokenize` | `wordpunct_tokenize` | `TreebankWordTokenizer` |
|---------|:-:|:-:|:-:|:-:|
| Splits Sentences | ✅ | ❌ | ❌ | ❌ |
| Splits Words | ❌ | ✅ | ✅ | ✅ |
| Handles Contractions | ❌ | ✅ | ❌ | ✅ |
| Regex Based | ❌ | ❌ | ✅ | ❌ |

### 📓 Notebooks
| File | Description |
|------|-------------|
| [`tokenization.ipynb`](tokenization/tokenization.ipynb) | Theory + all 4 tokenizers with examples |
| [`revision+theory-oftokenization.ipynb`](tokenization/revision+theory-oftokenization.ipynb) | Revision with Sundar Pichai corpus |
| [`practise.ipynb`](tokenization/practise.ipynb) | Practice exercises on tokenization |

---

## Chapter 2 — 🧹 Text Processing

> After tokenization, text must be **cleaned and normalized** before feeding it into any model. This chapter covers the core text preprocessing techniques.

📂 **Directory:** [`text_processing/`](text_processing/)

### 📘 What You'll Learn

#### 2.1 Stemming
Reduces words to their **root/base form** by chopping off suffixes. May produce non-dictionary words.

| Stemmer | Description | Example |
|---------|-------------|---------|
| **PorterStemmer** | Most common, rule-based | `"consulting"` → `"consult"` |
| **RegexpStemmer** | Custom regex patterns (`ing$\|s$\|e$`) | `"Following"` → `"Follow"` |
| **SnowballStemmer** | Improved Porter (supports multiple languages) | `"consulting"` → `"consult"` |

```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()
ps.stem("consulting")  # → 'consult'
```

#### 2.2 Lemmatization
Reduces words to their **dictionary form (lemma)** using vocabulary and morphological analysis. Always produces valid words.

```python
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
lem.lemmatize("went", pos='v')  # → 'go'
lem.lemmatize("has", pos='v')   # → 'have'
```

> 💡 **Stemming vs Lemmatization:** Stemming is faster but crude (`"better"` → `"better"`). Lemmatization is slower but accurate (`"better"` → `"good"` with POS tag).

#### 2.3 Stopwords Removal
Removes common words (`is`, `the`, `a`, `in`, etc.) that add no meaningful information.

```python
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
tokens = [word for word in tokens if word not in stop_words]
# Reduces 149 tokens → 105 tokens (29% reduction!)
```

#### 2.4 Parts of Speech (POS) Tagging
Assigns grammatical tags to each token (Noun, Verb, Adjective, etc.).

```python
from nltk.tag import pos_tag
pos_tag(tokens)
# → [('Pichai', 'NNP'), ('began', 'VBD'), ('career', 'NN'), ...]
```

| Tag | Meaning | Example |
|-----|---------|---------|
| `NNP` | Proper Noun | Google, Pichai |
| `VBD` | Past Tense Verb | began, joined |
| `NN` | Noun | career, engineer |
| `JJ` | Adjective | short, better |
| `IN` | Preposition | in, at, of |

#### 2.5 Named Entity Recognition (NER)
Identifies and classifies named entities (Person, Organization, Location, etc.) in text.

```python
from nltk.chunk import ne_chunk
ne_chunk(pos_tag(tokens)).draw()  # Visual tree of entities
```

### 📓 Notebooks
| File | Description |
|------|-------------|
| [`stemming.ipynb`](text_processing/stemming.ipynb) | Porter, Regexp, Snowball stemmers |
| [`lemmitazation.ipynb`](text_processing/lemmitazation.ipynb) | WordNet Lemmatizer with POS tags |
| [`stopword.ipynb`](text_processing/stopword.ipynb) | Stopwords removal + Lemmatization pipeline |
| [`parts-of-speach.ipynb`](text_processing/parts-of-speach.ipynb) | POS tagging with `pos_tag()` |
| [`name-entity-recognition.ipynb`](text_processing/name-entity-recognition.ipynb) | NER with `ne_chunk()` on Google corpus |

---

## Chapter 3 — 🎒 Bag of Words (BoW)

> **Bag of Words** converts text into a **numerical vector** by counting word frequencies — ignoring grammar and word order. It's like putting words in a bag where only their count matters.

📂 **Directory:** [`bow(bag of words)/`](bow(bag%20of%20words)/)

### 📘 What You'll Learn

#### 3.1 BoW Workflow
The complete text-to-vector pipeline:
```
Raw Text → Tokenization → Stopwords Removal → Lemmatization → CountVectorizer → Numerical Vectors
```

#### 3.2 CountVectorizer
Converts a collection of text documents into a matrix of token counts.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, binary=True)
X = cv.fit_transform(corpus).toarray()
```

| Parameter | What It Does |
|-----------|-------------|
| `max_features` | Keep only top N most frequent words |
| `binary=True` | 1 if word present, 0 if not (Binary BoW) |
| `binary=False` | Actual word frequency counts |
| `ngram_range=(1,3)` | Include unigrams, bigrams, and trigrams |

#### 3.3 N-Grams
Captures word context by considering sequences of N consecutive words.

| Type | Example from `"I love NLP"` |
|------|----------------------------|
| Unigram (1) | `"I"`, `"love"`, `"NLP"` |
| Bigram (2) | `"I love"`, `"love NLP"` |
| Trigram (3) | `"I love NLP"` |

#### 3.4 🚀 Project: Spam Classification
End-to-end spam detection using BoW + **Multinomial Naive Bayes**:
1. Text preprocessing (regex clean → lowercase → stemming → stopwords removal)
2. Feature extraction with `CountVectorizer` (Binary BoW + N-Grams)
3. Train-test split (80/20)
4. Classification with `MultinomialNB`
5. Evaluation with `classification_report` and `accuracy_score`

### 📓 Notebooks
| File | Description |
|------|-------------|
| [`workflow-of-bow/bow.ipynb`](bow(bag%20of%20words)/workflow-of-bow/bow.ipynb) | BoW workflow from scratch |
| [`implementation.ipynb`](bow(bag%20of%20words)/implementation.ipynb) | CountVectorizer + Binary BoW + N-Grams |
| [`spamclassification.ipynb`](bow(bag%20of%20words)/spamclassification.ipynb) | Spam detection project with Naive Bayes |

---

## Chapter 4 — 📊 TF-IDF

> **TF-IDF** (Term Frequency — Inverse Document Frequency) improves upon BoW by giving **higher weight to rare, important words** and lower weight to common ones.

📂 **Directory:** [`tf-idf/`](tf-idf/)

### 📘 What You'll Learn

#### 4.1 The Math Behind TF-IDF

| Component | Formula | Meaning |
|-----------|---------|---------|
| **TF** (Term Frequency) | `count(word) / total_words` | How often a word appears in a document |
| **IDF** (Inverse Document Frequency) | `log(total_docs / docs_containing_word)` | How rare/unique a word is across all documents |
| **TF-IDF** | `TF × IDF` | Balances frequency with uniqueness |

> 💡 Words that appear in **every** document get a low IDF score (like stopwords), while rare, meaningful words get a high score.

#### 4.2 TfidfVectorizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Basic TF-IDF
tfidf = TfidfVectorizer(max_features=10)
X = tfidf.fit_transform(corpus).toarray()

# TF-IDF with N-Grams
tfidf_ngram = TfidfVectorizer(max_features=30, ngram_range=(1, 2))
X_ngram = tfidf_ngram.fit_transform(corpus).toarray()
```

#### 4.3 BoW vs TF-IDF

| Feature | Bag of Words | TF-IDF |
|---------|:-:|:-:|
| Word Importance | Equal weight | Weighted by rarity |
| Common Words | High counts | Low scores |
| Rare Words | Low counts | High scores |
| Best For | Simple classification | Document relevance ranking |

### 📓 Notebooks
| File | Description |
|------|-------------|
| [`tf-idf.ipynb`](tf-idf/tf-idf.ipynb) | TfidfVectorizer implementation + N-Gram TF-IDF |

---

## Chapter 5 — 🧠 Word2Vec

> **Word2Vec** represents words as **dense vectors** in a high-dimensional space where semantically similar words are **close together**. Unlike BoW/TF-IDF, it captures **meaning and relationships** between words.

📂 **Files:** [`word2vec.ipynb`](word2vec.ipynb) · [`word2vec_projects.ipynb`](word2vec_projects.ipynb)

### 📘 What You'll Learn

#### 5.1 Loading Pre-trained Word2Vec
Using Google's pre-trained Word2Vec model (trained on 3 billion words from Google News):

```python
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')  # 300-dimensional vectors
vec_king = wv["king"]  # 300-dim vector for "king"
```

#### 5.2 Word Similarity
Measure how similar two words are (cosine similarity):

```python
wv.similarity('cricket', 'bat')  # → 0.374
```

#### 5.3 Vector Arithmetic ✨
The famous equation: **King − Man + Woman = Queen**

```python
vec = wv["man"] - wv["boy"] + wv["girl"]
wv.most_similar([vec])
# → [('woman', 0.87), ('man', 0.81), ('girl', 0.72), ...]
```

> 🎯 This shows Word2Vec captures **semantic relationships** — analogies like "man is to boy as woman is to girl" are encoded in the vector space.

### 📓 Notebooks
| File | Description |
|------|-------------|
| [`word2vec.ipynb`](word2vec.ipynb) | Word2Vec basics — embeddings, similarity, vector arithmetic |
| [`word2vec_projects.ipynb`](word2vec_projects.ipynb) | Word2Vec applied projects |

---

## 🗺️ NLP Pipeline — The Big Picture

```
                    ┌─────────────────────────────────────────────────┐
                    │            NLP Text Processing Pipeline         │
                    └─────────────────────────────────────────────────┘
                                          │
          ┌───────────────────────────────────────────────────────────┐
          │                                                           │
    ╔═══════════╗    ╔══════════════╗    ╔════════════════╗    ╔═══════════════╗
    ║   Step 1  ║    ║    Step 2    ║    ║     Step 3     ║    ║    Step 4     ║
    ║Tokenization║    ║ Text Process ║    ║ Vectorization  ║    ║   Modeling   ║
    ╚═══════════╝    ╚══════════════╝    ╚════════════════╝    ╚═══════════════╝
          │                 │                    │                     │
    ┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
    │sent_tok  │     │Stemming      │     │BoW           │     │Naive Bayes  │
    │word_tok  │     │Lemmatization │     │TF-IDF        │     │SVM          │
    │punct_tok │     │Stopwords     │     │Word2Vec      │     │Deep Learning│
    │Treebank  │     │POS / NER     │     │              │     │             │
    └─────────┘     └──────────────┘     └──────────────┘     └─────────────┘
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Programming Language |
| **NLTK** | Tokenization, Stemming, Lemmatization, POS, NER |
| **Scikit-Learn** | CountVectorizer, TfidfVectorizer, Naive Bayes, Metrics |
| **Gensim** | Word2Vec pre-trained models |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Regex (re)** | Text cleaning |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/atulshahi6310/nlp.git
cd nlp

# Install dependencies
pip install nltk scikit-learn gensim pandas numpy

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## 📂 Project Structure

```
nlp/
├── 📁 tokenization/
│   ├── tokenization.ipynb                    # Theory + 4 tokenizers
│   ├── revision+theory-oftokenization.ipynb  # Revision notes
│   └── practise.ipynb                        # Practice exercises
│
├── 📁 text_processing/
│   ├── stemming.ipynb                        # Porter, Regexp, Snowball
│   ├── lemmitazation.ipynb                   # WordNet Lemmatizer
│   ├── stopword.ipynb                        # Stopwords removal
│   ├── parts-of-speach.ipynb                 # POS Tagging
│   └── name-entity-recognition.ipynb         # NER
│
├── 📁 bow(bag of words)/
│   ├── workflow-of-bow/
│   │   └── bow.ipynb                         # BoW pipeline
│   ├── implementation.ipynb                  # CountVectorizer + N-Grams
│   └── spamclassification.ipynb              # 🚀 Spam Detection Project
│
├── 📁 tf-idf/
│   └── tf-idf.ipynb                          # TF-IDF + N-Gram TF-IDF
│
├── word2vec.ipynb                            # Word2Vec basics
├── word2vec_projects.ipynb                   # Word2Vec projects
├── spam.csv                                  # Dataset for classification
└── README.md                                 # 📖 This booklet
```

---

## 👨‍💻 Author

**Atul Kumar Shahi**
B.Tech CSE (AI/ML)

---

<p align="center">
  <b>⭐ Star this repo if you found it helpful!</b>
</p>
