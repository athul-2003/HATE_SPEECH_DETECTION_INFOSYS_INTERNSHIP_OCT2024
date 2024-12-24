# Hate Speech Detection and Twitter Text Classification

## Author
**N. Tejasri**  
**Mentor:** Dr. N. Jagan Mohan

---

## Project Overview
This project focuses on detecting hate speech and classifying Twitter texts using Natural Language Processing (NLP) techniques and Machine Learning. Inspired by the work of t-davidson, this project aims to improve upon the original research, introduce new features, and provide fresh analysis of the evolving challenges in hate speech detection.

## Problem Statement
Freedom of speech on the Internet has led to the misuse of social media platforms like Twitter, where hate speech and offensive language proliferate. This project addresses these challenges by identifying gaps in existing solutions and proposing enhancements through comprehensive research and innovative methods.

---

## Methodology
1. **Dataset**: 
   - Publicly available dataset from CrowdFlower.

2. **Text Pre-processing**: 
   - Punctuation removal, tokenization, stopword removal, stemming, and removal of URLs and mentions.

3. **Feature Engineering**: 
   - Extracted features include:
     - N-gram tf-idf weights
     - Sentiment polarity scores
     - Doc2Vec vector columns
     - Readability scores

4. **Machine Learning Models**:
   - Logistic Regression
   - Random Forest
   - Naive Bayes
   - Support Vector Machines (SVM)

5. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1-Score

---

## Results Obtained
- **Logistic Regression**: 
  - Performed consistently across most feature sets, with notable issues in feature set F7 for the "hate" label.

- **Random Forest Classifier**:
  - Demonstrated the best overall performance across all feature sets, except when tf-idf scores were excluded.

- **Naive Bayes Classifier**:
  - Less effective overall but performed better with feature set F7 compared to others.

- **SVM Classifier**:
  - Consistent performance but faced challenges with feature sets F4 and F7.

### Key Findings:
- **Most Important Feature**: 
  - TF-IDF scores significantly enhance classification accuracy.
- **Sentiment Scores**:
  - Essential for distinguishing between hate speech and offensive language.
- **Doc2Vec Columns**:
  - Found to have minimal impact on classification accuracy.
- **Overall Winner**:
  - Random Forest Classifier outperformed others in terms of accuracy and F1-Score.

---

## Summary
- The dataset was pre-processed to remove unwanted content and clean the text.
- Feature extraction included tf-idf weights, sentiment polarity scores, and readability scores.
- Various classification models were evaluated for performance against different feature sets.

**Challenges Addressed**:
- Differentiating hate speech from offensive language remains a complex task.
- Proposed features provide substantial improvements in detection.
- Results offer a valuable resource for mitigating toxic language on Twitter.

### Future Work
- Enhance feature extraction methods to address existing challenges.
- Conduct detailed error analysis to improve classification robustness.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```
4. Refer to the results and visualizations generated in the `/output` folder.

---

## Acknowledgments
Special thanks to Dr. N. Jagan Mohan for his mentorship and guidance throughout this project. This work builds upon the foundation laid by t-davidson's research.

