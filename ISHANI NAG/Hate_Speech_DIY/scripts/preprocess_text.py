import pandas as pd
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary NLTK data
nltk.download("stopwords")
from nltk.corpus import stopwords

# File paths
INPUT_FILE = "../data/hate_speech_data.csv"
PROCESSED_TRAIN_FILE = "../data/processed_train.csv"
PROCESSED_TEST_FILE = "../data/processed_test.csv"
VOCAB_SIZE = 10000  # Size of the vocabulary for Tokenizer
MAX_SEQUENCE_LENGTH = 100  # Max length for padded sequences


def clean_text(text):
    """
    Function to clean input text by:
    1. Removing special characters and numbers
    2. Removing stopwords
    """
    stop_words = set(stopwords.words("english"))
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text


def preprocess_data():
    """
    Function to preprocess the dataset:
    1. Load raw data
    2. Clean 'tweet' text column
    3. Tokenize text and pad sequences
    4. Save processed datasets
    """
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    # Check if required columns exist
    if "tweet" not in df.columns or "class" not in df.columns:
        raise ValueError("Input CSV must contain 'tweet' and 'class' columns.")

    # Clean text data
    print("Cleaning tweet text...")
    df["cleaned_tweet"] = df["tweet"].apply(clean_text)

    # Split into train-test
    print("Splitting into train and test sets...")
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Tokenize and pad text
    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["cleaned_tweet"])

    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_df["cleaned_tweet"])
    test_sequences = tokenizer.texts_to_sequences(test_df["cleaned_tweet"])

    # Pad sequences to uniform length
    train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    # Save processed data
    print("Saving processed datasets...")
    train_df = pd.DataFrame({"padded_sequences": list(train_padded), "class": train_df["class"].values})
    test_df = pd.DataFrame({"padded_sequences": list(test_padded), "class": test_df["class"].values})

    train_df.to_csv(PROCESSED_TRAIN_FILE, index=False)
    test_df.to_csv(PROCESSED_TEST_FILE, index=False)

    print("Preprocessing complete. Processed files saved.")


if __name__ == "__main__":
    preprocess_data()
