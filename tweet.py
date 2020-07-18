import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn
import os
import re
ps=PorterStemmer()
df=pd.read_csv('train1.csv')
processed_list=[]
for i in range(5):
    tweet=re.sub('[^a-zA-Z]',' ',df['tweet'][i])
    tweet=tweet.lower()
    tweet=tweet.split()
    tweet=[ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]
    tweet=' '.join(tweet)
    processed_list.append(tweet)
print(processed_list)
cv=CountVectorizer(max_features=5)
data=cv.fit_transform(processed_list)
x=data.toarray()
for i in range(5):
    y=df['label']

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2,random_state=10)
log_model=LogisticRegression()
log_model.fit(x,y)
predict=log_model.predict(x_test)
score=accuracy_score(y_test,predict)
print(score)
