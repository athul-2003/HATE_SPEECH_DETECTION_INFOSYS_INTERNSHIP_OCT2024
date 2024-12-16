import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load the dataset
dataset_path = '/content/modified_dataset.csv'
data = pd.read_csv(dataset_path)

# Assuming the dataset has 'text' and 'label' columns
texts = data['text'].astype(str).tolist()
labels = data['label'].tolist()
# Preprocess the text data
vocab_size = 10000  # Limit on the number of unique tokens
max_length = 100    # Maximum length of sequences
embedding_dim = 100 # Dimension of the embedding layer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
# Dynamically determine the number of classes and adjust labels to start from 0
num_classes = len(set(labels))
# Subtract 1 from all label values to make them start from 0
y_train = [label - 1 for label in y_train]
y_val = [label - 1 for label in y_val]
# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
# Define hyperparameters
lstm_units = 128     # Number of units in the LSTM layer
dropout_rate = 0.5   # Dropout rate for regularization
# Build the RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=lstm_units, return_sequences=False),  # LSTM layer
    Dropout(dropout_rate),                           # Dropout for regularization
    Dense(units=num_classes, activation='softmax')  # Output layer with softmax
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=3,  # Adjust the number of epochs as needed
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)



# Testing the model
# Prepare test data (assuming `X_val` and `y_val` are used as test data here)
predictions = model.predict(X_val)
predicted_classes = predictions.argmax(axis=1)
true_classes = y_val.argmax(axis=1)

# Predictions and true classes
print("Predicted Classes:", predicted_classes)
print("True Classes:", true_classes)



from sklearn.metrics import accuracy_score

# Assuming 'y_val' is the one-hot encoded true labels for the validation set
# and 'X_val' is the validation data
y_true = y_val.argmax(axis=1)  # Convert one-hot encoded labels to class indices
y_pred = model.predict(X_val).argmax(axis=1) # Get predicted class indices

accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")