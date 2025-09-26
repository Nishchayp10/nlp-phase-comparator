import streamlit as st
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Download necessary NLTK resources
# -------------------------------
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# Fix for different environments (old/new NLTK versions)
try:
    nltk.download("averaged_perceptron_tagger", quiet=True)
except:
    pass
try:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except:
    pass

lemmatizer = nltk.WordNetLemmatizer()

# -------------------------------
# Safe POS Tagging Wrapper
# -------------------------------
def safe_pos_tag(tokens):
    """Safely apply POS tagging, compatible with all NLTK versions."""
    try:
        return nltk.pos_tag(tokens)
    except LookupError:
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        return nltk.pos_tag(tokens)

# -------------------------------
# NLP Phase Functions
# -------------------------------

def lexical_processing(texts):
    """Lexical phase: cleaning + tokenization + lemmatization"""
    cleaned = []
    for t in texts:
        tokens = nltk.word_tokenize(str(t).lower())
        tokens = [lemmatizer.lemmatize(re.sub(r"[^a-z]", "", w)) for w in tokens if w.isalpha()]
        cleaned.append(" ".join(tokens))
    return cleaned

def semantic_processing(texts):
    """Semantic phase: POS tagging"""
    processed = []
    for t in texts:
        tokens = nltk.word_tokenize(str(t).lower())
        tagged = safe_pos_tag(tokens)
        words = [w for w, pos in tagged]  # keep only words, drop POS tags for simplicity
        processed.append(" ".join(words))
    return processed

def pragmatic_processing(texts):
    """Pragmatic phase: remove stopwords"""
    stopwords = set(nltk.corpus.stopwords.words("english"))
    processed = []
    for t in texts:
        tokens = nltk.word_tokenize(str(t).lower())
        tokens = [w for w in tokens if w not in stopwords]
        processed.append(" ".join(tokens))
    return processed

def synaptic_processing(texts):
    """Synaptic phase: character n-grams"""
    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3))
    return vectorizer, vectorizer.fit_transform(texts)

def discourse_integration_processing(texts):
    """Discourse integration phase: bigram word features"""
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    return vectorizer, vectorizer.fit_transform(texts)

# -------------------------------
# Model Training Function
# -------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    if model_name == "Naive Bayes":
        model = MultinomialNB()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "SVM":
        model = LinearSVC()
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    else:
        raise ValueError("Invalid model name")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="NLP Phases Comparison", layout="wide")

st.title("📊 NLP Phases Comparison with Machine Learning Models")
st.markdown(
    """
    This app allows you to:
    1. Upload your dataset (`.csv`)  
    2. Choose the **text column** and **label column**  
    3. Select a **machine learning model**  
    4. Compare performance across **different NLP phases**  
    """
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head())

    # Column selection
    text_col = st.selectbox("📝 Select the Text Column", df.columns)
    label_col = st.selectbox("🏷️ Select the Label Column", df.columns)

    # Model selection
    model_choice = st.selectbox(
        "🤖 Choose Machine Learning Model",
        ["Naive Bayes", "Decision Tree", "SVM", "Logistic Regression", "KNN"],
    )

    # Run button
    if st.button("🚀 Run Analysis"):
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(str).tolist()

        results = {}

        # --------------------------------------
        # Phase 1: Lexical
        # --------------------------------------
        try:
            st.info("🔎 Running Lexical Phase...")
            processed = lexical_processing(texts)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(processed)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            acc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)
            results["Lexical"] = acc
        except Exception as e:
            st.error(f"❌ Error while processing Lexical phase:\n{e}")

        # --------------------------------------
        # Phase 2: Semantic
        # --------------------------------------
        try:
            st.info("🧠 Running Semantic Phase...")
            processed = semantic_processing(texts)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(processed)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            acc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)
            results["Semantic"] = acc
        except Exception as e:
            st.error(f"❌ Error while processing Semantic phase:\n{e}")

        # --------------------------------------
        # Phase 3: Pragmatic
        # --------------------------------------
        try:
            st.info("💡 Running Pragmatic Phase...")
            processed = pragmatic_processing(texts)
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(processed)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            acc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)
            results["Pragmatic"] = acc
        except Exception as e:
            st.error(f"❌ Error while processing Pragmatic phase:\n{e}")

        # --------------------------------------
        # Phase 4: Synaptic
        # --------------------------------------
        try:
            st.info("🔤 Running Synaptic Phase...")
            vectorizer, X = synaptic_processing(texts)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            acc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)
            results["Synaptic"] = acc
        except Exception as e:
            st.error(f"❌ Error while processing Synaptic phase:\n{e}")

        # --------------------------------------
        # Phase 5: Discourse Integration
        # --------------------------------------
        try:
            st.info("🔗 Running Discourse Integration Phase...")
            vectorizer, X = discourse_integration_processing(texts)
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
            acc = train_and_evaluate(X_train, X_test, y_train, y_test, model_choice)
            results["Discourse Integration"] = acc
        except Exception as e:
            st.error(f"❌ Error while processing Discourse Integration phase:\n{e}")

        # --------------------------------------
        # Final Results
        # --------------------------------------
        st.subheader("✅ Accuracy Results by NLP Phase")
        st.write(results)

        if results:
            plt.figure(figsize=(8, 5))
            plt.bar(results.keys(), results.values(), color="skyblue")
            plt.ylabel("Accuracy")
            plt.xlabel("NLP Phase")
            plt.title(f"Model: {model_choice}")
            st.pyplot(plt)
