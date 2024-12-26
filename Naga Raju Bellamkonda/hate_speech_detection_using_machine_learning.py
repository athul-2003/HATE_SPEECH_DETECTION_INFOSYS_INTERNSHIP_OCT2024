#Hate speech detection using machine learning
import pandas as pd
import numpy as np
import nltk
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from nltk.util import pr
from nltk.corpus import stopwords

# Download the stopwords dataset
nltk.download('stopwords')

#import data set
data=pd.read_csv("labeled_data.csv")
print(data.head())

#Map the columns for hate speech
data['labels']=data['class'].map({0:"Hate Speech" , 1:"Offensive Language" , 2:"Normal"})
print(data.head())

data=data[['tweet','labels']]
print(data.head())

#clean the sentence in data set
def clean(text):
  text=str(text).lower()
  text=re.sub('\[.*?\]','',text)
  text=re.sub('https?://\S+|www\.\S+','',text)
  text=re.sub('<.*?>+','',text)
  text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
  text=re.sub('\n','',text)
  text=re.sub('\w*\d\w*','',text)
  text=[word for word in text.split(' ') if word not in stopwords.words('english')] # specify language for stopwords
  text=" ".join(text)
  return text

 # Apply clean to the 'tweet' column
data['tweet']=data['tweet'].apply(clean) 

#train data set
x=np.array(data['tweet'])
y=np.array(data['labels'])

cv=CountVectorizer()
x=cv.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)

# Now, for the SVM part:
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42) 

# SVM - works with sparse matrices
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Random Forest - works with sparse matrices
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# Convert sparse matrices to dense arrays for Naive Bayes
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

# Naive Bayes - requires dense arrays
nb_model = GaussianNB()
nb_model.fit(X_train_dense, y_train) 

# SVM
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# Naive Bayes
nb_predictions = nb_model.predict(X_test.toarray()) 
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)

# Random Forest
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)

#validate the dataset
sample="bitch you are so stupid"
data=cv.transform([sample]).toarray()
print(clf.predict(data))




--------
-----------------
#CONTACT#
NAME: BELLAMKONDA NAGA RAJU
EMAIL: bellnagaraju3307@gmail.com