#Hate Speech Detection using Machine Learning Model

!pip install nltk

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

tweet_df = pd.read_csv('labeled_data.csv')

tweet_df.head()

tweet_df.info()

print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")

def data_processing(tweet):
  tweet = tweet.lower()
  tweet = re.sub(r"https\S+|www\S+https\S+", '', tweet, flags=re.MULTILINE)
  tweet = re.sub(r'\@w+|\#','', tweet)
  tweet = re.sub(r'[^\w\s]','',tweet)
  tweet_tokens = word_tokenize(tweet)
  filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
  return " ".join(filtered_tweets)

tweet_df.tweet = tweet_df['tweet'].apply(data_processing)

tweet_df = tweet_df.drop_duplicates('tweet')

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
  tweet = [lemmatizer.lemmatize(word) for word in data]
  return data

tweet_df['tweet'] = tweet_df['tweet'].apply(lambda x: lemmatizing(x))

print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")

tweet_df.info()

tweet_df['tweet'].value_counts()

fig = plt.figure(figsize=(5,5))
sns.countplot(x='class',data=tweet_df)

fig = plt.figure(figsize=(7,7))
colors =("red","gold")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = tweet_df['class'].value_counts()
# Get the number of unique values in 'class' column
num_categories = len(tags)
# Create explode tuple with the correct length
explode = tuple([0.1] + [0] * (num_categories - 1))
tags.plot(kind='pie',autopct='%1.1f%%',shadow=True,colors=colors,startangle=90,wedgeprops=wp,explode=explode,label='')
plt.title('Distribution of sentiment')

non_hate_tweets = tweet_df[tweet_df['class'] == 0]  # Changed 'label' to 'class'
non_hate_tweets.head()

text = " ".join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15),facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non-hate tweets', fontsize=19)
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer # Import the TfidfVectorizer class from sklearn

vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet']) # Now you can use the TfidfVectorizer

feature_names = vect.get_feature_names_out() # Use get_feature_names_out() instead of get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))

vect = TfidfVectorizer(ngram_range=(1,3)).fit(tweet_df['tweet'])

feature_names = vect.get_feature_names_out() # Use get_feature_names_out() for scikit-learn versions 1.0 and above
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))

X = tweet_df['tweet']
Y = tweet_df['class']
X = vect.transform(X) # Changed 'transforms' to 'transform'

from sklearn.model_selection import train_test_split # Import train_test_split

X = tweet_df['tweet']
Y = tweet_df['class']
X = vect.transform(X) # Changed 'transforms' to 'transform'

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))

from sklearn.linear_model import LogisticRegression # Import LogisticRegression from sklearn.linear_model
from sklearn.metrics import accuracy_score # Import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))

from sklearn.metrics import confusion_matrix, classification_report # Import confusion_matrix and classification_report

print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(x_train, y_train) # Fit the model with training data
logreg_predict = logreg.predict(x_test)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()
plt.show()

from sklearn.experimental import enable_halving_search_cv  # Enable the experimental feature
from sklearn.model_selection import HalvingGridSearchCV  # Now you can import it
import warnings
warnings.filterwarnings('ignore')

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid ={'C':[100,10,1.0,0.1,0.001], 'solver' :['newton.cg','lbfgs','liblinear']}
# Now you can use GridSearchCV as intended:
grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 3)
grid.fit(x_train, y_train)

print("Best Cross validation score:{:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

y_pred = grid.predict(x_test)

logreg_acc =accuracy_score(y_pred, y_test)
print("Test accuracy:{:2f}%".format(logreg_acc*100))

# Assuming 'logreg' is the trained Logistic Regression model
y_train_pred = logreg.predict(x_train)  # Get predictions for the training data

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Print the training accuracy
print(f'Training Accuracy: {train_accuracy:.2f}')

print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))









--------
--------------
#*CONTACT*#
Name: BELLAMKONDA NAGA RAJU
Email: bellnagaraju3307@gmail.com