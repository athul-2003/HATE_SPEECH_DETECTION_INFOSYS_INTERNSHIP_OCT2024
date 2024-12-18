
# Hate Speech Detection Project

Welcome to the Hate Speech Detection Project, developed as part of the Infosys Spring Board Internship 5.0 by Soumyadeep Banerjee. This project explores the use of natural language processing (NLP) techniques and machine learning algorithms to identify hate speech and offensive content in social media text, specifically tweets.
________________________________________
📋 Introduction
Social media platforms have revolutionized communication, enabling billions of users to share thoughts and ideas instantly. However, this openness has also led to an increase in inappropriate content, including hate speech. Hate speech includes offensive discourse targeting individuals or groups based on race, religion, gender, or other inherent traits. Left unchecked, it can disrupt social harmony and propagate prejudice and violence.
Detecting hate speech manually is challenging given the vast scale of social media activity. This is where automated detection systems come into play. Using advanced machine learning and natural language processing techniques, this project aims to classify social media content into categories such as hate speech, offensive language, or neutral content.
________________________________________
📂 Dataset Overview
The project utilizes the Hate Speech and Offensive Language dataset, which consists of 25,296 tweets labeled into three categories:
1.	Hate Speech: Tweets containing hate speech.
2.	Offensive Language: Tweets that are offensive but do not qualify as hate speech.
3.	Neither: Neutral tweets that do not fall under the previous categories.
Dataset Columns:
•	count: Number of retweets or likes the tweet received.
•	hate_speech: Binary value (1/0) indicating the presence of hate speech.
•	offensive: Binary value (1/0) indicating offensive language.
•	neither: Binary value (1/0) indicating neither hate speech nor offensive content.
•	class: Categorical label representing the classification of the tweet.
•	tweet: The actual text of the tweet.
________________________________________
⚙️ Project Workflow
1. Data Preprocessing
•	Text cleaning (removing special characters, URLs, and unnecessary spaces).
•	Tokenization and normalization using NLTK.
2. Model Training
Built and evaluated models using machine learning algorithms (e.g., Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), Decision tree, SVM).
3. Evaluation Metrics
•	Accuracy: Measures overall performance.
•	Precision: Assesses performance for each category.
•	Confusion Matrix: Visualizes true positives, false positives, true negatives, and false negatives.
4. Visualization
•	Created plots to visualize class distributions, model performance, and confusion matrices using Matplotlib.
________________________________________
🛠️ Tools and Libraries
Python Libraries
•	Pandas: Data manipulation and analysis.
•	NumPy: Numerical computations.
•	Scikit-learn: Model training, testing, and evaluation.
•	Matplotlib: Data visualization.
•	NLTK: Text preprocessing and tokenization.
Core NLP Techniques
•	Tokenization
•	Stopword removal
•	Text classification
________________________________________
🚀 How to Use This Project
       Prerequisites
•	Python 3.8 or higher
•	Jupyter Notebook or any Python IDE
•	Required libraries installed via pip (pip install -r requirements.txt).
.
________________________________________
📊 Key Results: Obtained Model Metrics are as follows:
•	Logistic Regression:
•	Training Accuracy: 89.75%
•	Testing Accuracy: 86.85%
•	Random Forest:
•	Training Accuracy: 99.74%
•	Testing Accuracy: 89.49%
•	K-Nearest Neighbors:
•	Training Accuracy: 89.59%
•	Testing Accuracy: 85.72%
•	Decision tree:
•	Training Accuracy: 99.74%
•	Testing Accuracy: 89.44%
•	SVM:
•	Training Accuracy: 96.51%
•	Testing Accuracy: 88.83%________________________________________
🌟 Future Enhancements
•	Integrate a real-time hate speech detector for live social media monitoring.
•	Explore advanced NLP techniques, such as transformers (e.g., BERT), for improved accuracy.
________________________________________
📖 References
•	Understanding Hate Speech
•	Hate Speech Detection Task
•	Hate Speech Dataset
•	Toxicity Dataset
________________________________________
🧑‍💻 Contact
Soumyadeep Banerjee
mailto:soubanj270@gmail.com 
Feel free to reach out for questions, collaborations, or feedback on the project.


