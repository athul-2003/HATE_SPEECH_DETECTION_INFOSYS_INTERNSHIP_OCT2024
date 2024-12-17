import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# Load your preprocessed training and testing data
train_features_path = 'C:\\Users\\dell\\Desktop\\Infosys Springboard Project\\Hate_Speech_DIY\\scripts\\train_features.csv'
train_target_path = 'C:\\Users\\dell\\Desktop\\Infosys Springboard Project\\Hate_Speech_DIY\\scripts\\train_target.csv'
test_features_path = 'C:\\Users\\dell\\Desktop\\Infosys Springboard Project\\Hate_Speech_DIY\\scripts\\test_features.csv'
test_target_path = 'C:\\Users\\dell\\Desktop\\Infosys Springboard Project\\Hate_Speech_DIY\\scripts\\test_target.csv'

# Load train and test splits
X_train = pd.read_csv(train_features_path)
y_train = pd.read_csv(train_target_path).values.flatten()
X_test = pd.read_csv(test_features_path)
y_test = pd.read_csv(test_target_path).values.flatten()

# Data preprocessing:
# Ensure all input features are converted to numeric arrays and scaled if needed
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM (sequence modeling requires 3D data: [samples, time_steps, features])
# Reshape features to fit the LSTM's expected 3D input format
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# Create the LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Sigmoid since it's binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=10,  # Adjust this number for better training
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model performance
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the trained model in the 'models' directory
model_save_path = 'models/lstm_hate_speech_model.h5'
model.save(model_save_path)
print(f'Model saved successfully at {model_save_path}')
