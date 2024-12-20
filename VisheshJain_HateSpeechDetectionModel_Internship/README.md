------Hate Speech Detection with Machine Learning---------
    This project implements a Hate Speech Detection System using a Logistic Regression model. The system processes tweets, classifies them into hate speech or non-hate speech, and provides an interactive web interface using Gradio.

Name : Vishesh Mahesh Jain
Mentor : Dr. N. Jagan Mohan

------Features-------

Preprocessing of tweet data, including cleaning, tokenization, lemmatization, and removal of stop words.
Visualization of tweet data distribution and word frequency using word clouds and pie charts.
Feature extraction using TF-IDF Vectorization with n-grams.
Balanced dataset generation using SMOTE for better model performance.
Classification of tweets using Logistic Regression with hyperparameter tuning.
Model and vectorizer serialization for deployment.
Interactive Gradio interface for real-time prediction of hate speech.

----Requirements--------

The project requires the following libraries:
Python (>=3.7)
pandas
numpy
nltk
matplotlib
seaborn
wordcloud
scikit-learn
imbalanced-learn
pickle
gradio

---Dataset---
The dataset contains labeled tweets with the following columns:
tweet: The text of the tweet.
label: Binary labels (0 = Non-Hate Speech, 1 = Hate Speech).
It is loaded from a CSV file named train.csv.
-----Preprocessing----
Duplicate Removal: Removes duplicate tweets to ensure unique entries.
Text Cleaning:
Removes URLs, hashtags, mentions, special characters, and emojis.
Converts text to lowercase.
Tokenization: Splits the text into individual words.
Stop Words Removal: Eliminates common stop words (e.g., "is", "the").
Lemmatization: Reduces words to their base forms.
-----Visualization----
Label Distribution: A bar chart and pie chart show the proportion of hate and non-hate tweets.
Word Clouds: Visualize the most common words in both hate and non-hate tweets.
----Model Building------
Feature Extraction
TF-IDF Vectorization: Converts text into numerical vectors with support for 1-gram, 2-gram, and 3-gram combinations.
-------Dataset Balancing-----
SMOTE (Synthetic Minority Oversampling Technique): Handles class imbalance by oversampling the minority class.
Classification
Logistic Regression: A simple yet effective machine learning algorithm.
Hyperparameter tuning using GridSearchCV to find the optimal model parameters.
Metrics
Accuracy: Overall model performance.
Confusion Matrix: Displays true positives, true negatives, false positives, and false negatives.
Classification Report: Shows precision, recall, and F1-score.
Deployment
Serialization
The trained model and vectorizer are saved as .pkl files using pickle.
Gradio Interface
An interactive web interface allows users to input a tweet and classify it as hate speech or non-hate speech.
Example:
Input: I hate you
Output: This is a Hate Speech Tweet!


------Acknowledgments---------
The dataset used in this project is publicly available and specifically designed for hate speech detection.
Libraries like Scikit-learn, NLTK, and Gradio were crucial for the implementation of this project.

-------Contact-----
For any questions or suggestions, feel free to reach out:
Vishesh Jain