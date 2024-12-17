#LOGISTIC REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv('labeled_data.csv')

# Display the first few rows of the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Summary statistics of the data
print(data.describe())

# Check the data types
print(data.dtypes)

# Load dataset
data = pd.read_csv('labeled_data.csv')

# Assuming your target column is named 'class' instead of 'target'
X = data.drop(columns=['class'])  # Change 'target' to 'class'
y = data['class']              # Change 'target' to 'class'

# Define the features (X) and the target (y)
X = data.drop(columns=['class'])
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the features (X) and the target (y)
X = data[['count', 'hate_speech', 'offensive_language', 'neither']]  # Select only numerical features
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

import seaborn as sns

# Plot confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load a sample dataset for classification (Logistic Regression)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (classification target)

# Split dataset into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)

# Train the model
log_reg.fit(X_train, y_train)

# Training Accuracy
train_pred = log_reg.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print(f'Logistic Regression - Training Accuracy: {train_accuracy * 100:.2f}%')

# Testing Accuracy
test_pred = log_reg.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f'Logistic Regression - Testing Accuracy: {test_accuracy * 100:.2f}%')







--------
--------------
#*CONTACT*#
Name: BELLAMKONDA NAGA RAJU
Email: bellnagaraju3307@gmail.com