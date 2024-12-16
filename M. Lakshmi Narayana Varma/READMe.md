# Hate Speech Detection  

## Introduction  
The Hate Speech Detection project is designed to find and classify harmful language that targets people or groups based on their race, gender, religion, or other characteristics. used Deep Learning Model (LSTM) and machine learning models to analyze text from social media and other online platforms to detect hate speech effectively
## Steps Performed  
1. **Data Preprocessing**:  
   - Removed missing values.  
   - Cleaned text data by converting to lowercase, removing punctuation, stopwords, numbers, and extra whitespaces.  
   - Applied stemming and lemmatization for text standardization.  

2. **Class Balancing**:  
   - Handled class imbalance using Random Oversampling with the RandomOverSampler technique to ensure an equal representation of both classes.  

3. **Feature Extraction**:  
   - Utilized TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text into numerical inputs for models.  

4. **Model Training & Evaluation**:  
   - Data was split into training ,testing and validation sets.  
   - Trained several machine learning and deep learning models, evaluating their performance using precision, recall, F1-score, and accuracy.  

## Models Used  
- Gradient Boosting  
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- LSTM (Long Short-Term Memory Network)  

## Model Comparison Results  

### Gradient Boosting:  
- Precision: 0.92  
- Recall: 0.92  
- F1 Score: 0.92  
- Accuracy: 92%  

### Logistic Regression:  
- Precision: 0.99  
- Recall: 0.99  
- F1 Score: 0.99  
- Accuracy: 99%  

### Random Forest:  
- Precision: 0.99  
- Recall: 0.99  
- F1 Score: 0.99  
- Accuracy: 99%  

### Support Vector Machine (SVM):  
- Precision: 0.99  
- Recall: 1.00  
- F1 Score: 1.00  
- Accuracy: 99.58%  

### LSTM (Epochs 1-5):  

| Epoch | Training Accuracy | Validation Accuracy | Loss  | Validation Loss |  
|-------|--------------------|---------------------|-------|-----------------|  
| 1     | 79.45%            | 92.92%             | 0.5369| 0.2403          |  
| 2     | 97.01%            | 97.29%             | 0.1027| 0.0770          |  
| 3     | 99.48%            | 98.23%             | 0.0240| 0.0562          |  
| 4     | 99.84%            | 98.12%             | 0.0102| 0.0708          |  
| 5     | 99.84%            | 98.23%             | 0.0066| 0.0583          |  

- **Test Accuracy**: 98.23%  

### LSTM with Best Hyperparameters  

**Hyperparameters**:  
- Embedding Dimension: 228  
- Spatial Dropout Rate: 0.4  
- LSTM Units: 128  
- LSTM Dropout: 0.2  
- Recurrent Dropout: 0.2  
- Learning Rate: 0.0016  

| Epoch | Training Accuracy | Validation Accuracy | Loss  | Validation Loss |  
|-------|--------------------|---------------------|-------|-----------------|  
| 1     | 81.12%            | 94.58%             | 0.4168| 0.1574          |  
| 2     | 98.57%            | 98.12%             | 0.0499| 0.0609          |  
| 3     | 99.74%            | 97.92%             | 0.0119| 0.0725          |  
| 4     | 99.82%            | 98.33%             | 0.0064| 0.0521          |  
| 5     | 99.95%            | 98.54%             | 0.0033| 0.0596          |  

- **Test Accuracy**: 98.54%  

## Conclusion  

- **Support Vector Machine (SVM)** delivered the highest accuracy (**99.58%**), precision, and recall, making it the best-performing model.  
- Logistic Regression followed closely with a high F1 Score (**99%**) and accuracy.  
- LSTM achieved excellent results with fine-tuned hyperparameters, producing a test accuracy of **98.54%**.  
- Gradient Boosting and Random Forest models performed well but were outperformed by SVM and LSTM.  
