# 📢 Hate Speech Detection with LSTM Model

This project implements a Long Short-Term Memory (LSTM) neural network for classifying hate speech in text data using natural language processing (NLP) techniques.

---

## 🚀 **Project Overview**

The goal of this project is to build a machine learning pipeline capable of detecting hate speech from textual data. We use LSTM, a type of recurrent neural network (RNN), to train and predict patterns in sequence data.

---

## 🛠️ **Features**

- Preprocessing of raw text data to remove unnecessary tokens.
- Training an LSTM-based model for text classification.
- Data balancing using SMOTE to address class imbalances.
- Model evaluation with both training and testing accuracy metrics.
- Saved models for later use with `.h5` or `.keras` formats.
- Visualization of training/testing performance over epochs.

---

## 📊 **Dataset**

The dataset consists of labeled text data to classify sentences or phrases as either:
- **Hate Speech** (target class: `1`)
- **Non-Hate Speech** (target class: `0`)

Dataset sources may include:
- Manually labeled datasets
- Standard NLP hate speech repositories

---

## 🏆 **Model Architecture**

The model is an LSTM-based sequence classifier with the following key features:
1. **LSTM Layers** for sequence modeling.
2. **Dropout** layers for regularization to avoid overfitting.
3. Dense output layer with sigmoid activation for binary classification (hate vs. non-hate).

---

## ⚙️ **Setup Instructions**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hate_speech_detection.git
cd hate_speech_detection
```

---

### 2. Install Required Dependencies
Make sure you have Python 3.x installed. Install dependencies with:
```bash
pip install -r requirements.txt
```

---

### 3. Download Necessary Data
The `nltk_download.py` script is included to ensure dependencies like tokenization or stopword processing are available:
```bash
python scripts/nltk_download.py
```

---

### 4. Preprocess Data
Run preprocessing scripts to clean and prepare your text data.
```bash
python scripts/preprocess_text.py
```

---

### 5. Balance Dataset with SMOTE
To balance the dataset using oversampling techniques:
```bash
python scripts/SMOTE.py
```

---

### 6. Train the Model
Train the LSTM model with the following command:
```bash
python scripts/train_model.py
```

This will:
- Train the LSTM model.
- Save the model to `models/` as `lstm_hate_speech_model.h5` or `.keras`.

---

### 7. Evaluate the Model
You can evaluate the trained model performance on unseen test data using:
```bash
python scripts/split.py
```

---

## 🏗️ **Model Training Details**

### Architecture
The LSTM architecture consists of:
- LSTM layers
- Dropout for regularization
- Dense layers for classification.

#### Model Summary
```
Total Params: ~1,310,369
Trainable params: ~1,310,369
```

---

## 🖥️ **Model Deployment**

You can save and use this model for deployment purposes. If needed, use Flask or FastAPI to integrate this model into a web application.

---

## 🛠️ **Repository Contents**

### 📂 **Data**
- `hate_speech_data.csv`: Raw hate speech dataset.
- `hate_speech_data.p`: Serialized data for preprocessed input.

### 📂 **Models**
- `lstm_hate_speech_model.h5`: Saved trained LSTM model.

### 📂 **Scripts**
- `nltk_download.py`: Ensure all NLTK data is downloaded.
- `preprocess_text.py`: Preprocesses raw text data for tokenization and cleaning.
- `SMOTE.py`: Balances dataset using SMOTE.
- `train_model.py`: Trains the LSTM model.
- `split.py`: Splits dataset into training and testing sets and handles evaluation.

### 📂 **Notebooks**
- Used for initial exploratory analysis and experiments.

### 📂 **Results**
- Model metrics or saved results after training.

---

## 🏆 **Results**

### Final Test Accuracy:
- **Training Accuracy:** ~99%
- **Testing Accuracy:** ~94%