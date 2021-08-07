# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:46:08 2021

@author: HP
"""


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

df = pd.read_csv('airline_sentiment_analysis.csv')
df = df.iloc[:, [1,2]]

df.airline_sentiment.value_counts()

sentences = []
for index, row in df.iterrows():
    sentences.append(row["text"])
      
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
add_stop_words = ['@', '_', '!', '#', '$', '%', '^', '&', '*', '(', ')', '<', '>', '?', '/', '|', '}', '{', '~', ':', '`', '..']
stop_words.extend(add_stop_words)

document = []
for i in range(0, len(sentences)):
    sentence = sentences[i]
    sentence = ''.join([j for j in sentence if not j.isdigit()])
    word_tokens = word_tokenize(sentence)
    sentence = [w for w in word_tokens if not w.lower() in set(stop_words)]
    sentence = ' '.join(sentence)
    document.append(sentence)

lems = []
wnl = WordNetLemmatizer()
for doc in document:
  list2 = nltk.word_tokenize(doc)
  lemmatized_string = ' '.join([wnl.lemmatize(words) for words in list2]) 
  lems.append(lemmatized_string)

words_list = []
for doc in lems:
    words_list.append(re.findall('(\\w+)', doc.lower()))
   
itisha = []
for word in words_list:
    subitisha = ' '.join([words for words in word])
    itisha.append(subitisha)

df['clean_text'] = itisha

from sklearn.feature_extraction.text import TfidfVectorizer
Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
X = Tfidf.fit_transform(itisha).toarray()

Y = df.iloc[:, 0:1].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y[:, 0])

kf = StratifiedKFold(10, shuffle=True, random_state=42)

fold = 0
for train, test in kf.split(X, df['airline_sentiment']):  
    fold+=1
    print(f"Fold #{fold}")
        
    x_train = X[train]
    y_train = Y[train]
    x_test = X[test]
    y_test = Y[test]

    from sklearn.linear_model import LogisticRegression
    classifier1 = LogisticRegression()
    classifier1.fit(x_train, y_train)
    
    from sklearn.neighbors import KNeighborsClassifier
    classifier2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier2.fit(x_train, y_train)
    
    from sklearn.svm import SVC
    classifier3 = SVC(kernel='rbf')
    classifier3.fit(x_train, y_train)
    
    from sklearn.naive_bayes import GaussianNB
    classifier4 = GaussianNB()
    classifier4.fit(x_train, y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    classifier5 = DecisionTreeClassifier(criterion='entropy')
    classifier5.fit(x_train, y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    classifier6 = RandomForestClassifier(n_estimators=10, criterion='entropy')
    classifier6.fit(x_train, y_train)
    
    from xgboost import XGBClassifier
    classifier7 = XGBClassifier()
    classifier7.fit(x_train, y_train)
    
    y_hat1 = classifier1.predict(x_test)
    y_hat2 = classifier2.predict(x_test)
    y_hat3 = classifier3.predict(x_test)
    y_hat4 = classifier4.predict(x_test)
    y_hat5 = classifier5.predict(x_test)
    y_hat6 = classifier6.predict(x_test)
    y_hat7 = classifier7.predict(x_test)
    
    print("Logistic Regression accuracy:" +  str(accuracy_score(list(y_test), list(y_hat1))))
    print(confusion_matrix(list(y_test), list(y_hat1)))
    print("Logistic Regression F1 score: " + str(f1_score(y_test, y_hat1, average='weighted')))

    print("KNeighbors Classifier accuracy:" +  str(accuracy_score(list(y_test), list(y_hat2))))
    print(confusion_matrix(list(y_test), list(y_hat2)))
    print("KNeighbors Classifier F1 score: " + str(f1_score(y_test, y_hat2, average='weighted')))

    print("SVC accuracy:" +  str(accuracy_score(list(y_test), list(y_hat3))))
    print(confusion_matrix(list(y_test), list(y_hat3)))
    print("SVC F1 score: " + str(f1_score(y_test, y_hat3, average='weighted')))

    print("GaussianNB accuracy:" +  str(accuracy_score(list(y_test), list(y_hat4))))
    print(confusion_matrix(list(y_test), list(y_hat4)))
    print("GaussianNB F1 score: " + str(f1_score(y_test, y_hat4, average='weighted')))

    print("Decision Tree Classifier accuracy:" +  str(accuracy_score(list(y_test), list(y_hat5))))
    print(confusion_matrix(list(y_test), list(y_hat5)))
    print("Decision Tree Classifier F1 score: " + str(f1_score(y_test, y_hat5, average='weighted')))

    print("Random Forest Classifier accuracy:" +  str(accuracy_score(list(y_test), list(y_hat6))))
    print(confusion_matrix(list(y_test), list(y_hat6)))
    print("Random Forest Classifier F1 score: " + str(f1_score(y_test, y_hat6, average='weighted')))

    print("XgBoost Classifier accuracy:" +  str(accuracy_score(list(y_test), list(y_hat7))))
    print(confusion_matrix(list(y_test), list(y_hat7)))
    print("XgBoost Classifier F1 score: " + str(f1_score(y_test, y_hat7, average='weighted')))