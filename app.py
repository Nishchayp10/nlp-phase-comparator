"""
streamlit_nlp_models_app.py

Updated Streamlit app (large / full version) that:
- Implements five ML classifiers (NB, Decision Tree, SVM, Logistic, KNN)
- Implements five NLP phases (Lexical, Semantic, Synaptic, Pragmatic, Discloser Integration)
- Uses NLTK for lemmatization, POS tagging and sentiment (VADER) BUT avoids nltk.word_tokenize
  (so it won't trigger punkt/punkt_tab lookup errors on Streamlit)
- Uses a regex tokenizer and a safe POS tagger wrapper to avoid averaged_perceptron_* lookup issues
- Trains on user-uploaded CSV or /mnt/data/politifact_full.csv if present
- Visualizes accuracy per phase and detailed reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy import sparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------ NLTK downloads (safe) ------------------
# We keep required downloads but avoid tokenizers that cause punkt_tab lookups by not using word_tokenize.
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Attempt to ensure POS taggers are available under either name (some environments expect _eng)
try:
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    pass
try:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception:
    pass

# ------------------ Globals ------------------
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ------------------ Safe utilities ------------------
def safe_pos_tag(tokens):
    """POS tagger that retries by downloading tagger resources if necessary."""
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        # try to download both possible names then retry
        nltk.download('averaged_perceptron_tagger', quiet=True)
        try:
            return nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            return nltk.pos_tag(tokens)

def get_wordnet_pos(treebank_tag: str):
    """Map POS tag to WordNet POS for lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text: str) -> str:
    """Lowercase, remove urls, html tags, keep letters and spaces, collapse whitespace."""
    text = str(text or "")
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)  # only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def regex_tokenize(text: str):
    """Lightweight tokenizer that does not depend on punkt.
    Returns a list of lowercased alphabetic tokens (no stopwords)"""
    text = clean_text(text)
    if not text:
        return []
    # split by whitespace because clean_text removed non-letters already
    tokens = text.split()
    # filter stopwords and single-letter tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# ------------------ Tokenizers used by vectorizers ------------------
def simple_tokenize(text: str):
    """Used for lexical-phase vectorizers: basic cleaning + stopword removal."""
    return regex_tokenize(text)

def lemma_tokenize(text: str):
    """Lemmatize tokens using WordNet POS tags (avoid using word_tokenize)."""
    text = clean_text(text)
    if not text:
        return []
    tokens = text.split()
    pos_tags = safe_pos_tag(tokens)
    lemmas = []
    for token, pos in pos_tags:
        wn_pos = get_wordnet_pos(pos)
        lemma = lemmatizer.lemmatize(token, pos=wn_pos)
        if lemma not in STOPWORDS and len(lemma) > 1:
            lemmas.append(lemma)
    return lemmas

def pos_tokenize(text: str):
    """Return tokens annotated with POS tag (e.g. 'word_NN') for syntactic features."""
    text = clean_text(text)
    if not text:
        return []
    tokens = text.split()
    pos_tags = safe_pos_tag(tokens)
    tokens_with_pos = [f"{w}_{p}" for w, p in pos_tags if w not in STOPWORDS and len(w) > 1]
    return tokens_with_pos

def lemma_synonym_tokenize(text: str):
    """Lemmatize and add (at most) one synonym per token to do light semantic expansion."""
    tokens = lemma_tokenize(text)
    expanded = []
    for token in tokens:
        expanded.append(token)
        try:
            syns = wordnet.synsets(token)
            if syns:
                for lemma in syns[0].lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name != token:
                        # add only one synonym to avoid blowing up vocabulary
                        expanded.append(name)
                        break
        except Exception:
            pass
    return expanded

# ------------------ sklearn Transformers ------------------
class SentimentTransformer(BaseEstimator, TransformerMixin):
    """Create numeric sentiment/document features with VADER + simple stats."""
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for doc in X:
            s = self.sid.polarity_scores(str(doc))
            compound_scaled = (s['compound'] + 1.0) / 2.0  # 0..1
            words = re.findall(r"[a-z]+", str(doc).lower())
            doc_len = len(words)
            avg_word_len = np.mean([len(w) for w in words]) if doc_len > 0 else 0.0
            feats.append([s['neg'], s['neu'], s['pos'], compound_scaled, doc_len, avg_word_len])
        return sparse.csr_matrix(np.array(feats))

class TextSelector(BaseEstimator, TransformerMixin):
    """If pipeline input is a DataFrame, select a column by key and return its values."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept numpy array or pandas series too
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X[self.key].astype(str).values
        else:
            return np.array(X).astype(str)

# ------------------ classifier factory ------------------
def get_classifier(name: str):
    name = name.lower()
    if 'naive' in name:
        return MultinomialNB()
    elif 'decision' in name or 'tree' in name:
        return DecisionTreeClassifier(random_state=42)
    elif 'svc' in name or 'svm' in name or 'support' in name:
        return LinearSVC(max_iter=10000, random_state=42)
    elif 'logistic' in name:
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    elif 'knn' in name or 'nearest' in name:
        return KNeighborsClassifier()
    else:
        raise ValueError(f"Unknown classifier: {name}")

# ------------------ pipelines per NLP phase ------------------
def build_pipeline(phase: str, classifier_name: str):
    clf = get_classifier(classifier_name)
    # When we pass a tokenizer to vectorizer, set token_pattern=None and lowercase=False
    if phase == 'Lexical':
        vect = CountVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False)
        return Pipeline([('vect', vect), ('clf', clf)])

    elif phase == 'Semantic':
        vect = TfidfVectorizer(tokenizer=lemma_synonym_tokenize, token_pattern=None, lowercase=False)
        return Pipeline([('vect', vect), ('clf', clf)])

    elif phase == 'Synaptic':
        vect = TfidfVectorizer(tokenizer=pos_tokenize, token_pattern=None, lowercase=False, ngram_range=(1,2), max_features=50000)
        return Pipeline([('vect', vect), ('clf', clf)])

    elif phase == 'Pragmatic':
        union = FeatureUnion([
            ('tfidf', TfidfVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False, max_features=50000)),
            ('sent', SentimentTransformer())
        ])
        return Pipeline([('union', union), ('clf', clf)])

    elif phase == 'Discloser Integration':
        union = FeatureUnion([
            ('lex', TfidfVectorizer(tokenizer=simple_tokenize, token_pattern=None, lowercase=False, max_features=30000)),
            ('sem', TfidfVectorizer(tokenizer=lemma_synonym_tokenize, token_pattern=None, lowercase=False, max_features=30000)),
            ('syn', TfidfVectorizer(tokenizer=pos_tokenize, token_pattern=None, lowercase=False, ngram_range=(1,2), max_features=30000)),
            ('sent', SentimentTransformer())
        ])
        return Pipeline([('union', union), ('clf', clf)])
    else:
        raise ValueError(f"Unknown NLP phase: {phase}")

# ------------------ evaluation ------------------
def evaluate_phase(phase, classifier_name, X_train, X_test, y_train, y_test):
    pipe = build_pipeline(phase, classifier_name)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    return {
        'pipeline': pipe,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'preds': preds
    }

# ------------------ Streamlit UI ------------------
def main():
    st.set_page_config(page_title='NLP Phase vs ML Model Comparator', layout='wide')
    st.title('Compare NLP Phases across ML Models (NLTK only — no punkt)')
    st.markdown(
        """
        Upload your CSV, pick the text column and label column, choose one ML algorithm,
        and the app will train the selected algorithm using five different NLP pre-processing pipelines (phases).
        The result shows accuracy per phase and plots a comparison chart.
        """
    )

    with st.sidebar:
        st.header('Options')
        uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
        sample_n = st.number_input('Max rows to use (0 = all)', min_value=0, value=5000, step=500)
        classifier_name = st.selectbox('Choose ML algorithm', [
            'Naive Bayes Classification',
            'Decision Tree Classification',
            'Support Vector Machine',
            'Logistic Regression',
            'K - Nearest Neighbour'
        ])
        run_button = st.button('Run / Process')

    # load dataframe
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        # try to load default path (helpful if running in environment with file present)
        default_path = '/mnt/data/politifact_full.csv'
        try:
            df = pd.read_csv(default_path)
            st.info(f"No upload detected — loaded default CSV from {default_path}")
        except Exception:
            st.info('Upload a CSV file (or place politifact_full.csv at /mnt/data when running locally).')

    if df is not None:
        st.subheader('Data preview')
        st.write('Shape: ', df.shape)
        st.dataframe(df.head())

        # column selection
        cols = df.columns.tolist()
        text_col = st.selectbox('Text column (input)', cols, index=0)
        label_col = st.selectbox('Label column (target)', cols, index=min(1, len(cols)-1))

        if run_button:
            # prepare data
            data = df[[text_col, label_col]].dropna()
            if sample_n and sample_n > 0:
                data = data.sample(min(sample_n, len(data)), random_state=42)
            data = data.reset_index(drop=True)

            X = data[text_col].astype(str)
            y_raw = data[label_col]

            # label encode if needed
            le = LabelEncoder()
            try:
                y = le.fit_transform(y_raw.astype(str))
            except Exception:
                y = y_raw

            if len(np.unique(y)) < 2:
                st.error('Need at least two classes in the target label to train classifiers.')
                return

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            phases = ['Lexical', 'Semantic', 'Synaptic', 'Pragmatic', 'Discloser Integration']
            results = {}

            progress = st.progress(0)
            total = len(phases)
            for i, phase in enumerate(phases, start=1):
                with st.spinner(f'Training {phase} pipeline with {classifier_name}...'):
                    try:
                        res = evaluate_phase(phase, classifier_name, X_train, X_test, y_train, y_test)
                        results[phase] = res
                    except Exception as e:
                        st.error(f'Error while processing phase {phase}: {e}')
                        results[phase] = {'accuracy': 0.0, 'error': str(e)}
                progress.progress(int(i/total*100))

            # show accuracies
            accs = {phase: float(results[phase]['accuracy']) if 'accuracy' in results[phase] else 0.0 for phase in phases}
            st.subheader('Accuracy comparison across NLP phases')
            acc_df = pd.DataFrame.from_dict(accs, orient='index', columns=['accuracy']).sort_values('accuracy', ascending=False)
            st.table(acc_df)

            # bar chart
            st.bar_chart(acc_df['accuracy'])

            # show details per phase
            for phase in phases:
                st.markdown('---')
                st.subheader(f'{phase} — Details')
                item = results[phase]
                if 'error' in item:
                    st.error(f"Phase failed: {item['error']}")
                    continue
                st.write(f"Accuracy: {item['accuracy']:.4f}")

                report = item['report']
                report_df = pd.DataFrame(report).transpose()
                st.write('Classification report:')
                st.dataframe(report_df)

                cm = item['confusion_matrix']
                st.write('Confusion matrix:')
                st.write(cm)

    st.markdown('---')
    st.markdown('**Notes & tips**')
    st.markdown(
        """
        - This version avoids `nltk.word_tokenize` so it will not request the `punkt_tab` resource.
        - If your dataset is very large, increase the 'Max rows to use' or run the app locally with more RAM.
        - For faster runs while testing, reduce 'Max rows to use' to e.g. 2000 or 1000.
        - If you want me to further optimize (add caching, persist trained models, or add per-class metrics), tell me which feature you'd like next.
        """
    )

if __name__ == '__main__':
    main()
