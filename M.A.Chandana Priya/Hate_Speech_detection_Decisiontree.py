import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
# Training Path
def train_model(training_path):
    # Load the training dataset
    train_data = pd.read_csv(training_path)
    # Assuming the dataset has 'text' and 'label' columns
    X_train = train_data['text']
    y_train = train_data['label']
    # Text Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Train Decision Tree Classifier Model
    model = DecisionTreeClassifier()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer
# Paths to datasets
training_path = "/content/modified_dataset.csv"
# Train the model
model, vectorizer = train_model(training_path)


# Testing Path
def test_model(testing_path, model, vectorizer):
    # Load the testing dataset
    test_data = pd.read_csv(testing_path)
    # Assuming the dataset has 'text' and 'label' columns
    X_test = test_data['text']
    y_test = test_data['label']
    # Transform test data using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    # Predict on the test data
    y_pred = model.predict(X_test_tfidf)
    return y_test, y_pred


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
dataset_path = "/content/modified_dataset.csv"  
data = pd.read_csv(dataset_path)
# Assuming the dataset has 'text' and 'label' columns
texts = data['text'].values
labels = data['label'].values
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Create a TF-IDF vectorizer to convert text to numerical features
vectorizer = TfidfVectorizer(max_features=1000) 
# Fit the vectorizer to the training data and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transform the test data using the fitted vectorizer
X_test_tfidf = vectorizer.transform(X_test)
# Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
# Train the model using the TF-IDF features
dt_classifier.fit(X_train_tfidf, y_train)
# Make predictions on the test set using the TF-IDF features
y_pred = dt_classifier.predict(X_test_tfidf)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Generate and print the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n", report)