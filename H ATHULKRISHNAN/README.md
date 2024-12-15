# Hate Speech Detection

## Introduction
The Hate Speech Detection project focuses on identifying and categorizing language that incites hate or violence against individuals or groups based on factors such as race, religion, gender, or sexual orientation. This is achieved by using natural language processing (NLP) and machine learning methods to analyze text data from various online sources, including social media platforms and forums.

## Steps Performed
1. **Data Preprocessing**: 
   - Removed missing values.
   - Cleaned the text data by converting it to lowercase, removing punctuation, stopwords, numbers, and extra whitespaces.
   - Applied stemming and lemmatization to standardize words.

2. **Class Balancing**:
   - Used SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance and ensure equal representation of both classes.

3. **Feature Extraction**:
   - Used TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert text data into numerical features for model input.

4. **Model Training & Evaluation**:
   - Split the data into training and testing sets.
   - Trained multiple machine learning models and evaluated their performance using cross-validation and performance metrics.

## Models Used
- **Logistic Regression**
- **Random Forest Classifier**
- **K-Nearest Neighbors**
- **XGBoost**
- **AdaBoost**

## Results

| Model                  | Training Accuracy | Testing Accuracy | Precision | Recall  | F1 Score |
|------------------------|-------------------|------------------|-----------|---------|----------|
| Logistic Regression     | 94.72%            | 93.07%           | 93.56%    | 92.58%  | 93.07%   |
| Random Forest           | 88.38%            | 87.16%           | 94.05%    | 79.47%  | 86.15%   |
| K-Nearest Neighbors     | 92.14%            | 86.25%           | 94.42%    | 77.19%  | 84.94%   |
| XGBoost                | 89.13%            | 88.53%           | 95.81%    | 80.70%  | 87.61%   |
| AdaBoost               | 86.08%            | 86.05%           | 94.23%    | 76.95%  | 84.72%   |

### Conclusion
- **Logistic Regression** achieved the highest testing accuracy at **93.07%**.
- **XGBoost** exhibited the best **precision** (95.81%) while **Logistic Regression** had the highest **F1 Score** (93.07%).

This project demonstrates the effective application of machine learning techniques to classify hate speech from text data.

