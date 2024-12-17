from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the preprocessed dataset
import pandas as pd

# Assuming the dataset has been preprocessed and saved
data = pd.read_csv('processed_dataset.csv')

# Features and target
X = data['tweet']  # The text column
y = data['class']  # The target label: 0, 1, or 2

# Split data into train-test before resampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text data into numerical features using tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize
max_vocab_size = 10000
max_length = 50
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Apply SMOTE to resample the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_pad, y_train)

# Check class distribution after resampling
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())
