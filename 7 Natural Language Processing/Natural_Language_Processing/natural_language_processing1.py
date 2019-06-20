#Natural Language processing

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t', quoting=3)

#cleaning the process
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    reviews = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    reviews = reviews.lower()
    reviews = reviews.split()
    ps = PorterStemmer()
    reviews = [ps.stem(word) for word in reviews if not word in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    corpus.append(reviews)

#creating the bag of wors model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#splitting the dataset into traning and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#fitting the NAive bayes to training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)

#MAking the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)