
# Hate Speech Detection

Welcome to the **Hate Speech Detection Project**, developed as part of the **Infosys Spring Board Internship 5.0** by **Venkata Ramana Panigrahi**. This project explores the use of natural language processing (NLP) techniques and machine learning algorithms to identify hate speech and offensive content in social media text, specifically tweets.

---

## 📋 *Introduction*

Social media platforms have revolutionized communication, enabling billions of users to share thoughts and ideas instantly. However, this openness has also led to an increase in inappropriate content, including **hate speech**. Hate speech includes offensive discourse targeting individuals or groups based on race, religion, gender, or other inherent traits. Left unchecked, it can disrupt social harmony and propagate prejudice and violence.

Detecting hate speech manually is challenging given the vast scale of social media activity. This is where **automated detection systems** come into play. Using advanced machine learning and natural language processing techniques, this project aims to classify social media content into categories such as hate speech, offensive language, or neutral content.

---

## 📂 *Dataset Overview*

The project utilizes the **Hate Speech and Offensive Language** dataset, which consists of 25,296 tweets labeled into three categories:

1. **Hate Speech**: Tweets containing hate speech.
2. **Offensive Language**: Tweets that are offensive but do not qualify as hate speech.
3. **Neither**: Neutral tweets that do not fall under the previous categories.

### **Dataset Columns:**
- **count**: Number of retweets or likes the tweet received.
- **hate_speech**: Binary value (1/0) indicating the presence of hate speech.
- **offensive**: Binary value (1/0) indicating offensive language.
- **neither**: Binary value (1/0) indicating neither hate speech nor offensive content.
- **class**: Categorical label representing the classification of the tweet.
- **tweet**: The actual text of the tweet.

---

## ⚙️ **Project Workflow**

### **1. Data Preprocessing**
- Text cleaning (removing special characters, URLs, and unnecessary spaces).
- Tokenization and normalization using **NLTK**.
- Feature extraction using **TF-IDF** to convert text into numerical vectors.

### **2. Handling Class Imbalance**
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset, ensuring better performance across all classes.

### **3. Model Training**
- Built and evaluated models using **machine learning algorithms** (e.g., Logistic Regression, Random Forest) and **deep learning models** (e.g., LSTM, CNN) using **TensorFlow**.

### **4. Evaluation Metrics**
- **Accuracy**: Measures overall performance.
- **Precision, Recall, F1-Score**: Assesses performance for each category.
- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, and false negatives.

### **5. Visualization**
- Created plots to visualize class distributions, model performance, and confusion matrices using **Matplotlib**.

---

## 🛠️ **Tools and Libraries**

### **Python Libraries**
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Model training, testing, and evaluation.
- **Matplotlib**: Data visualization.
- **NLTK**: Text preprocessing and tokenization.
- **TensorFlow**: Deep learning framework.
- **Imbalanced-learn**: Handling class imbalance with oversampling techniques.

### **Core NLP Techniques**
- Tokenization
- Stopword removal
- TF-IDF vectorization
- Text classification

---

## 🚀 **How to Use This Project**

### **1. Prerequisites**
- Python 3.8 or higher
- Jupyter Notebook or any Python IDE
- Required libraries installed via pip (`pip install -r requirements.txt`).

### **2. Running the Project**
1. Clone this repository:
   ```bash
    https://github.com/Vishal101503/Hate_Speech_Detector.git
   ```
2. Navigate to the project directory:
   ```bash
   cd hate-speech-detection
   ```
3. Run the Jupyter Notebook or Python script to preprocess the data, train the model, and evaluate its performance.

---

## 📊 **Key Results**

- Achieved **95% accuracy** on test data using a Logistic Regression model.
- Improved classification for minority classes (hate speech) using **SMOTE**.
- Demonstrated the utility of **TF-IDF vectorization** for text-based feature extraction.

---

## 🌟 **Future Enhancements**

- Integrate a **real-time hate speech detector** for live social media monitoring.
- Explore advanced NLP techniques, such as **transformers (e.g., BERT)**, for improved accuracy.
- Fine-tune deep learning models to better understand context and nuance in text.
- Extend detection to multiple languages using multilingual datasets.

---

## 📖 **References**

- [Understanding Hate Speech](https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech)
- [Hate Speech Detection Task](https://paperswithcode.com/task/hate-speech-detection)
- [Hate Speech Dataset](https://paperswithcode.com/dataset/hate-speech-and-offensive-language)
- [Toxicity Dataset](https://github.com/nicknochnack/CommentToxicity)

---

## 🧑‍💻 **Contact**
**Venkata Ramana Panigrahi**  
📧 **vishalpanigrahi1015@gmail.com**  


Feel free to reach out for questions, collaborations, or feedback on the project.  

Happy Coding! 🎉
