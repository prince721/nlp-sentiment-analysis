import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ', df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review=' '.join(review)
    #print(review)
    corpus.append(review)
    #print(corpus[i])
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
Y=df.iloc[:,1].values




#spliting into training and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size= 0.2,random_state=0)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)