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

## Model Comparison Results

### Results

| Model                  | Training Accuracy | Testing Accuracy | Precision | Recall  | F1 Score |
|------------------------|-------------------|------------------|-----------|---------|----------|
| Logistic Regression     | 96.22%            | 94.11%           | 93.03%    | 95.36%  | 94.18%   |
| Random Forest           | 92.57%            | 90.37%           | 89.79%    | 91.07%  | 90.43%   |
| K-Nearest Neighbors     | 99.49%            | 67.59%           | 60.73%    | 99.44%  | 75.40%   |
| XGBoost                | 89.36%            | 88.80%           | 95.34%    | 81.56%  | 87.92%   |
| AdaBoost               | 88.28%            | 88.47%           | 95.13%    | 81.07%  | 87.54%   |

### Conclusion

- **Logistic Regression** achieved the highest testing accuracy at **94.11%**.
- **XGBoost** exhibited the best **precision** (95.34%) while **Logistic Regression** had the highest **F1 Score** (94.18%).
- **Random Forest** performed consistently well with a good balance between **precision** and **recall**, showing **90.43%** for F1 Score.
- **K-Nearest Neighbors** had the highest **training accuracy** (99.49%), but its **testing accuracy** was much lower (67.59%), indicating potential overfitting.
- **AdaBoost** had a balanced performance with good **precision** (95.13%) and **recall** (81.07%).

### Best Model Based on Testing Accuracy:
- **Model**: Logistic Regression
- **Training Accuracy**: 96.22%
- **Testing Accuracy**: 94.11%
- **Precision**: 93.03%
- **Recall**: 95.36%
- **F1 Score**: 94.18%


This project demonstrates the effective application of machine learning techniques to classify hate speech from text data.

