Hate Speech Detection Project
    This project uses machine learning to detect and classify text as Hate Speech or Not Hate Speech. The pipeline involves data cleaning, handling imbalanced datasets, and training a classification model.

Features
    Preprocessing of text data to remove noise.
    Handling class imbalance through upsampling.
    Visualization of class distributions.
    Classification using a machine learning pipeline with CountVectorizer, TfidfTransformer, and SGDClassifier.

Prerequisites
    Python (>=3.6)
    Libraries: pandas, matplotlib, seaborn, scikit-learn, numpy

Instructions to Run
Prepare Dataset:
    Ensure the presence of train.csv and test.csv files in the working directory.
    The training dataset should have three columns: id, label, and tweet.
    The test dataset should have two columns: id and tweet.
    
Run the Notebook:
    Open the Google Colab or Python script containing the code.
    Execute the cells sequentially.
Steps Performed:

Data Cleaning:
    Lowercase transformation and removal of special characters, mentions, URLs, and retweets.
Visualization:
    Bar charts show the class distribution before and after handling imbalances.
Handling Imbalanced Data:
    Upsample the minority class to balance the dataset.
Training:
    Split the data into training and testing sets.
    Train a pipeline with SGDClassifier on the upsampled dataset.
Evaluation:
    Predict on the test set and evaluate using the F1-score.
Output:
    Visualizations of class distribution.
    F1-score of the model performance on the test set.

Example Results
Class Distribution:
    Before Upsampling:
        Not Hate Speech: Majority
        Hate Speech: Minority
    After Upsampling:
        Both classes are balanced.
Model F1-Score: ~0.97.