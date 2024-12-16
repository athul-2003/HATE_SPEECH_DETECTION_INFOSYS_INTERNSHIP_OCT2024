import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return model, vectorizer
# Paths to datasets
training_path = "/content/modified_dataset.csv"
# Train the model
model, vectorizer = train_model(training_path)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Testing Path
def test_model(testing_path, model, vectorizer):
    test_data = pd.read_csv(testing_path)
    # Assuming the dataset has 'text' and 'label' columns
    X_test = test_data['text']
    y_test = test_data['label']
    # Transform test data using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)
    # Predict on the test data
    y_pred = model.predict(X_test_tfidf)
    return y_test, y_pred
# Paths to datasets
testing_path = "/content/modified_dataset.csv"
# Test the model
y_test, y_pred = test_model(testing_path, model, vectorizer)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
# Accuracy Path
def evaluate_model(y_test, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred)
    return accuracy, report
# Paths to datasets
testing_path = "/content/modified_dataset.csv"
# Test the model
y_test, y_pred = test_model(testing_path, model, vectorizer)
# Evaluate the model
accuracy, report = evaluate_model(y_test, y_pred)
# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
