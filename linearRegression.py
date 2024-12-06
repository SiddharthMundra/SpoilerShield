import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix, roc_auc_score
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
from multiprocessing import Pool, cpu_count
import ssl
from tqdm import tqdm  # For progress bar

# Ensure SSL compatibility for downloading resources
ssl._create_default_https_context = ssl._create_unverified_context

# -------------------------------
# Download NLTK Stopwords Once
# -------------------------------
nltk.download('stopwords')

# Initialize Stemmer and Stopwords Globally
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------------------
# Text Preprocessing Functions
# -------------------------------
def initializer():
    """
    Initializer function for multiprocessing.
    Initializes global variables in each subprocess.
    """
    global stemmer
    global stop_words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # Removed the line causing the error

def preprocess_text(text):
    """
    Preprocess a single sentence.
    """
    global stemmer
    global stop_words

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def preprocess_text_batch(texts):
    """
    Preprocess a batch of sentences using multiprocessing for efficiency.
    """
    print(f"Preprocessing {len(texts)} sentences...")

    # Use tqdm to show progress bar
    with Pool(processes=cpu_count(), initializer=initializer) as pool:
        chunksize = 1000
        total = len(texts)
        with tqdm(total=total, desc="Preprocessing", unit="sentence") as pbar:
            processed_texts = []
            for result in pool.imap(preprocess_text, texts, chunksize=chunksize):
                processed_texts.append(result)
                pbar.update()
    print("Batch preprocessing completed.")
    return processed_texts

# -------------------------------
# Feature Extraction (TF-IDF)
# -------------------------------
def vectorize_tfidf(X_train, X_test):
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        max_features=10000   # More features for richer representation
    )
    X_train_tfidf = vectorizer.fit_transform(tqdm(X_train, desc="Fitting TF-IDF Vectorizer"))
    X_test_tfidf = vectorizer.transform(tqdm(X_test, desc="Transforming Test Data"))
    print("Vectorization completed.")
    return X_train_tfidf, X_test_tfidf, vectorizer

# -------------------------------
# Train and Evaluate TF-IDF Model
# -------------------------------
def train_and_evaluate_tfidf(X_train, X_test, y_train, y_test):
    print("\n=== Evaluating TF-IDF Model ===")
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    print("Making predictions on the test set...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    print("Predictions completed.")

    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    roc_auc = roc_auc_score(y_test, y_probs)

    print(f"Precision: {precision:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

# -------------------------------
# Main Execution
# -------------------------------
def main():
    warnings.filterwarnings("ignore")
    nltk.download('stopwords')

    # Load and preprocess data
    file_path = 'balanced_goodreads_reviews.json'
    print("Loading dataset...")
    data = pd.read_json(file_path, lines=True)
    print(f"Dataset loaded with {len(data)} reviews.")

    sentences = []
    labels = []
    print("Extracting sentences and labels...")
    for idx, review in enumerate(tqdm(data['review_sentences'], desc="Processing Reviews")):
        for sentence in review:
            labels.append(sentence[0])
            sentences.append(sentence[1])
    data = pd.DataFrame({'sentence': sentences, 'is_spoiler': labels})
    print(f"Total sentences: {len(data)}")

    # Optionally, sample a subset for testing
    # data = data.sample(n=100000, random_state=42).reset_index(drop=True)

    # Preprocess text data
    print("Starting preprocessing of text data...")
    data['clean_sentence'] = preprocess_text_batch(data['sentence'].tolist())
    print("Text preprocessing completed.")

    # Balance dataset
    print("Balancing the dataset...")
    minority_class = data[data['is_spoiler'] == 1]
    majority_class = data[data['is_spoiler'] == 0].sample(len(minority_class), random_state=42)
    balanced_data = pd.concat([minority_class, majority_class]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Balanced dataset size: {len(balanced_data)}")

    X = balanced_data['clean_sentence']
    y = balanced_data['is_spoiler']
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print("Data splitting completed.")

    # Vectorize and evaluate
    X_train_tfidf, X_test_tfidf, vectorizer_tfidf = vectorize_tfidf(X_train, X_test)
    train_and_evaluate_tfidf(X_train_tfidf, X_test_tfidf, y_train, y_test)

if __name__ == "__main__":
    main()