#Lowercasing and Tokenization
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
#Ensure necessary NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
file_path = '/content/modified_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the path.")
# Check if 'text' column exists
if 'text' not in df.columns:
    print("Error: The dataset must contain a 'text' column.")
else:
     # Lowercasing and tokenization
    df['text_lower'] = df['text'].str.lower()
    df['tokens'] = df['text_lower'].apply(word_tokenize)
     # Display the processed DataFrame
    print(df.head())
    # Save the processed dataset to a new CSV
    df.to_csv('processed_dataset.csv', index=False)
    print("Processed dataset saved to 'processed_dataset.csv'.")


#Removing punctuations and stop words
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
file_path = '/content/modified_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the path.")
#Check if 'text' column exists
if 'text' not in df.columns:
    print("Error: The dataset must contain a 'text' column.")
else:
     # Function to remove punctuation
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    # Load stop words
    stop_words = set(stopwords.words('english'))
    # Function to remove stop words
    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in stop_words]
    # Processing steps
    df['text_clean'] = df['text'].str.lower()  # Lowercasing
    df['text_clean'] = df['text_clean'].apply(remove_punctuation)  # Remove punctuation
    df['tokens'] = df['text_clean'].apply(word_tokenize)  # Tokenization
    df['tokens_no_stopwords'] = df['tokens'].apply(remove_stopwords)  # Remove stop words
     # Display the processed DataFrame
    print(df[['text', 'tokens_no_stopwords']].head())
    # Save the processed dataset to a new CSV
    df.to_csv('processed_dataset.csv', index=False)
    print("Processed dataset saved to 'processed_dataset.csv'.")


#Stemming and lemmatization
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
 #Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') # For extended WordNet support
file_path = '/content/modified_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the path.")
# Check if 'text' column exists
if 'text' not in df.columns:
    print("Error: The dataset must contain a 'text' column.")
else:
     # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Function to apply stemming
    def apply_stemming(tokens):
        return [stemmer.stem(word) for word in tokens]
    # Function to apply lemmatization
    def apply_lemmatization(tokens):
        return [lemmatizer.lemmatize(word) for word in tokens]
    # Preprocessing steps
    df['text_lower'] = df['text'].str.lower() # Lowercasing
    df['tokens'] = df['text_lower'].apply(word_tokenize)  # Tokenization
    df['stemmed'] = df['tokens'].apply(apply_stemming)   # Apply stemming
    df['lemmatized'] = df['tokens'].apply(apply_lemmatization) # Apply lemmatization
     # Display the processed DataFrame
    print(df[['text', 'stemmed', 'lemmatized']].head())
    # Save the processed dataset to a new CSV
    df.to_csv('processed_dataset.csv', index=False)
    print("Processed dataset saved to 'processed_dataset.csv'.")



#Removing numbers,white spaces and special characters
import pandas as pd
import re
file_path = '/content/modified_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the path.")
# Check if 'text' column exists
if 'text' not in df.columns:
    print("Error: The dataset must contain a 'text' column.")
else:
     # Function to remove numbers, white spaces, and special characters
    def clean_text(text):
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = re.sub(r'\s+', ' ', text) # Remove extra white spaces
        text = re.sub(r'[^\w\s]', '', text) # Remove special characters
        return text.strip() # Remove leading/trailing white spaces
    # Apply cleaning
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Display the processed DataFrame
    print(df[['text', 'cleaned_text']].head())
     # Save the processed dataset to a new CSV
    df.to_csv('processed_dataset.csv', index=False)
    print("Processed dataset saved to 'processed_dataset.csv'.")


#Text Normalization
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
file_path = '/content/modified_dataset.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please check the path.")
# Check if 'text' column exists
if 'text' not in df.columns:
    print("Error: The dataset must contain a 'text' column.")
else:
    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Function for text normalization
    def normalize_text(text):
        text = text.lower() # Lowercase text
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = re.sub(r'[^\w\s]', '', text) # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        tokens = word_tokenize(text)  # Tokenize text
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # Lemmatization and stop word removal
        return ' '.join(tokens)  # Combine tokens back to a normalized string
    # Apply normalization to the 'text' column
    df['normalized_text'] = df['text'].apply(normalize_text)
    # Display the processed DataFrame
    print(df[['text', 'normalized_text']].head())
     # Save the processed dataset to a new CSV
    df.to_csv('processed_dataset.csv', index=False)
    print("Processed dataset saved to 'processed_dataset.csv'.")


#Tf-idf
!pip install nltk

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
dataset_path ="/content/modified_dataset.csv"
# Function to remove special characters, numbers, and extra white spaces
def preprocess_text(text):
    # Remove all non-alphanumeric characters (except spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Remove extra white spaces
    text = ' '.join(text.split())
    # Tokenize the text
    tokens = word_tokenize(text.lower())  # Lowercase during tokenization
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    # Apply stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    return ' '.join(lemmatized_tokens)
#Read the dataset
df = pd.read_csv(dataset_path)
# Check if 'label' column exists (you will create 'processed_text')
if 'label' not in df.columns:
    raise ValueError("The dataset must contain a 'label' column.")
# Apply preprocessing to create the 'processed_text' column
df['processed_text'] = df['text'].apply(preprocess_text)
# Initialize the TF-IDF Vectorizer with max_features=5000
vectorizer = TfidfVectorizer(max_features=5000)
# Apply fit_transform on the 'processed_text' column
X = vectorizer.fit_transform(df['processed_text']).toarray()
# Get the target variable (Label)
y = df['label'].values
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)