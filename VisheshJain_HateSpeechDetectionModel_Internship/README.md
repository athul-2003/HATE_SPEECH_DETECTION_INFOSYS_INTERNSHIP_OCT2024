
                                                Hate Speech Detection with Machine Learning
Name : Vishesh Mahesh Jain
Mentor : Dr. N. Jagan Mohan
Title : Hate Speech Detection with Machine Learning


This project implements a Hate Speech Detection System using a Logistic Regression model. The system processes tweets, classifies them as hate speech or non-hate speech, and provides an interactive web interface powered by Gradio.

Features
Comprehensive Data Preprocessing:
Cleaning, tokenization, lemmatization, and removal of stop words.
Visualization Tools:
Displays tweet data distribution and word frequency through word clouds and pie charts.
Feature Engineering:
TF-IDF Vectorization with support for n-grams (1-gram, 2-gram, 3-gram).
Dataset Balancing:
Uses SMOTE (Synthetic Minority Oversampling Technique) for improved model performance.
Machine Learning Model:
Logistic Regression with hyperparameter tuning using GridSearchCV.
Model Serialization:
Saves the trained model and vectorizer for deployment using pickle.
Interactive Interface:
Gradio-powered interface for real-time tweet classification.
Requirements
The project uses the following libraries:

Python (>=3.7)
pandas, numpy
nltk, matplotlib, seaborn
wordcloud, scikit-learn
imbalanced-learn
pickle, gradio
  
Dataset
The dataset contains labeled tweets and is loaded from a CSV file named train.csv. It includes the following columns:

tweet: The text of the tweet.
label: Binary classification labels (0 = Non-Hate Speech, 1 = Hate Speech).
Preprocessing Steps
Duplicate Removal: Ensures unique entries by removing duplicates.
Text Cleaning:
Removes URLs, hashtags, mentions, special characters, and emojis.
Converts text to lowercase.
Tokenization: Splits tweets into individual words.
Stop Word Removal: Removes common words like "is" and "the."
Lemmatization: Reduces words to their base forms for consistency.
Visualization
Label Distribution:
Bar charts and pie charts display proportions of hate and non-hate tweets.
Word Clouds:
Visualize frequent words in hate speech and non-hate speech categories.
Model Building
Feature Extraction
TF-IDF Vectorization: Converts text into numerical vectors using n-grams for richer contextual understanding.
Dataset Balancing
SMOTE: Addresses class imbalance by oversampling the minority class.
Classification
Logistic Regression:
A simple yet effective classifier.
Optimized using GridSearchCV for hyperparameter tuning.
Performance Metrics
Accuracy: Evaluates overall model performance.
Confusion Matrix: Highlights true positives, true negatives, false positives, and false negatives.
Classification Report: Summarizes precision, recall, and F1-score for each class.
Deployment
Serialization
Saves the trained model and TF-IDF vectorizer as .pkl files using pickle.
Interactive Gradio Interface
A user-friendly web interface for real-time tweet classification.
Example:
Input: "I hate you"
Output: "This is a Hate Speech Tweet!"

Acknowledgments
The dataset used in this project is publicly available and tailored for hate speech detection.
Special thanks to libraries like Scikit-learn, NLTK, and Gradio for enabling smooth implementation.


Contact
For any questions or suggestions, feel free to reach out:
Vishesh Jain



