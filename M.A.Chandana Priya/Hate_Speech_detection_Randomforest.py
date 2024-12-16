# Step 1: Train the Random Forest model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
file_path = "/content/modified_dataset.csv"
df = pd.read_csv(file_path)
# Check if the dataset has 'text' and 'label' columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")
# Preprocess text and labels
texts = df['text'].values
labels = df['label'].values # Assume 1: Hate Speech, 2: Offensive, 3: Neither
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42) # Using a linear kernel
svm_model.fit(X_train_tfidf, y_train)
print("Model Training Complete!")
df = pd.read_csv(file_path)
# Check if the dataset has 'text' and 'label' columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")
# Preprocess text and labels
texts = df['text'].values
labels = df['label'].values # Assume 1: Hate Speech, 2: Offensive, 3: Neither
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # 100 trees in the forest
rf_model.fit(X_train_tfidf, y_train)

print("Model Training Complete!")


# Step 2: Testing the Model (Making Predictions)
# Transform the test data using the fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)
# Make predictions
y_pred = rf_model.predict(X_test_tfidf)

print("Predictions Complete!")


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
file_path = "/content/modified_dataset.csv"
df = pd.read_csv(file_path)
# Assuming 'text' column contains the text data and 'label' column contains the labels
X = df['text']
y = df['label']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data using the fitted vectorizer
# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_tfidf, y_train)
# Make predictions on the test set
y_pred = rf_model.predict(X_test_tfidf)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest model: {accuracy * 100:.2f}%")
# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
# Visualize the confusion matrix using seaborn and matplotlib
class_names = ['Hate Speech', 'Offensive', 'Neither']  # Replace with your actual class names
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix for Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))