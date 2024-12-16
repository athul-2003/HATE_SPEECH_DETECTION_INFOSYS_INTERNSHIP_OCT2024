from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer
# Assuming your dataset is in a pandas DataFrame called 'df'
# with columns 'text' for the text data and 'label' for the labels
# 1. Preprocess text and labels (if not already done)
texts = df['text'].values
labels = df['label'].values
# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# 3. Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data using the fitted vectorizer
# 4. Create and train the SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)
print("SVM Model Training Complete!")

# Step 2: Testing and Finding Accuracy
# Transform the test data using the fitted vectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
X_test_tfidf = vectorizer.transform(X_test)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)
# Make predictions
y_pred = svm_model.predict(X_test_tfidf)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#visualization
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
file_path = "/content/modified_dataset.csv"
df = pd.read_csv(file_path)
# Ensure dataset has 'text' and 'label' columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")
# Preprocess text and labels
texts = df['text'].values
labels = df['label'].values # Assume 1: Hate Speech, 2: Offensive, 3: Neither
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000) # Adjust max_features as needed
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train an SVM classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
# Predict on test data
y_pred = svm_model.predict(X_test_tfidf)
# Dimensionality reduction using PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_tfidf.toarray())
# Plot the results
plt.figure(figsize=(10, 8))
colors = {1: 'blue', 2: 'red', 3: 'green'}
labels_text = {1: "Hate Speech", 2: "Offensive Language", 3: "Neither"}
for label in [1, 2, 3]:
    indices = y_pred == label
    plt.scatter(X_test_pca[indices, 0], X_test_pca[indices, 1], c=colors[label], label=labels_text[label], alpha=0.6, s=50)
# Add plot details
plt.title("SVM Results Visualization", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.legend()
plt.grid(alpha=0.4)
plt.show()
