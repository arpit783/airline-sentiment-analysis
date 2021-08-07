# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:36:24 2021

@author: HP
"""

import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
from keras import regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score

class SentimentAnalysisModel:
    def __init__(self, datafile = "airline_sentiment_analysis.csv"):
        self.df = pd.read_csv(datafile)
        
    def preprocess(self):
        sentences = []
        for index, row in self.df.iterrows():
            sentences.append(row["text"])
              
        stop_words = stopwords.words('english')
        add_stop_words = ['@', '_', '!', '#', '$', '%', '^', '&', '*', '(', ')', '<', '>', '?', '/', '|', '}', '{', '~', ':', '`', '..']
        stop_words.extend(add_stop_words)
        
        document = []
        for i in range(0, len(sentences)):
            sentence = sentences[i]
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
           
        text = []
        for word in words_list:
            subtext = ' '.join([words for words in word])
            text.append(subtext)
        
        self.df['clean_text'] = text
        
    def modelSelection(self, modelType):
        self.df['l'] = self.df['clean_text'].apply(lambda x: len(str(x).split(' ')))
        sequence_length = self.df.l.max()
        max_features = 20000 # this is the number of words we care about
        tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
        tokenizer.fit_on_texts(self.df['clean_text'].values)
        
        # this takes our sentences and replaces each word with an integer
        self.X = tokenizer.texts_to_sequences(self.df['clean_text'].values)
        
        # we then pad the sequences so they're all the same length (sequence_length)
        self.X = pad_sequences(self.X, sequence_length)
        
        
        Y = self.df.iloc[:, 0:1].values
        labelencoder = LabelEncoder()
        self.y = labelencoder.fit_transform(Y[:, 0])
        
        if(modelType=='randomEmbeddings'):
            embedding_dim = 200 # Kim uses 300 here
            num_filters = 100
            sequence_length = self.df.l.max()
            max_features = 20000 # this is the number of words we care about
            inputs = Input(shape=(sequence_length,), dtype='int32')
        
            # use a random embedding for the text
            embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        
            reshape = Reshape((sequence_length, embedding_dim, 1))(embedding_layer)
        
            # Note the relu activation which Kim specifically mentions
            # He also uses an l2 constraint of 3
            # Also, note that the convolution window acts on the whole 200 dimensions - that's important
            conv_0 = Conv2D(num_filters, kernel_size=(3, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)
            conv_1 = Conv2D(num_filters, kernel_size=(4, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)
            conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)
            
            # perform max pooling on each of the convoluations
            maxpool_0 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0)
            maxpool_1 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1)
            maxpool_2 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2)
            
            # concat and flatten
            concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
            flatten = Flatten()(concatenated_tensor)
            
            # do dropout and predict
            dropout = Dropout(0.5)(flatten)
            output = Dense(1, activation='sigmoid')(dropout)
            
            self.model = Model(inputs=inputs, outputs=output)
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif(modelType=='LogisticRegression'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = LogisticRegression()
        elif(modelType=='KNeighborsClassifier'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        elif(modelType=='svc'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = SVC(kernel='rbf')
        elif(modelType=='NaiveBayes'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = GaussianNB()
        elif(modelType=='DecisionTree'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = DecisionTreeClassifier(criterion='entropy')
        elif(modelType=='RandomForest'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = RandomForestClassifier(n_estimators=10, criterion='entropy')
        elif(modelType=='XgBoost'):
            Tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=0.001)
            self.X = Tfidf.fit_transform(self.df['clean_text'].values).toarray()
            self.model = XGBClassifier()
                    
    def splitAndTraining(self, folds, category):
        kf = StratifiedKFold(folds, shuffle=True, random_state=42)

        fold = 0
        for train, test in kf.split(self.X, self.df['airline_sentiment']):  
            fold+=1
            print(f"Fold #{fold}")
        
            x_train = self.X[train]
            y_train = self.y[train]
            self.x_test = self.X[test]
            self.y_test = self.y[test]
            
            if(category==1):
                self.model.fit(x_train, y_train, epochs=15, batch_size=64, verbose=1, validation_data = (self.x_test, self.y_test), shuffle=True)
            else:
                self.model.fit(x_train, y_train)
            
            break
                
    def predict(self, input_value):
        if input_value == None:
            y_hat = self.model.predict(self.x_test)
        else: 
            y_hat = self.model.predict(np.array([input_value]))
            
        for i in range(0, len(y_hat)):
            if(y_hat[i]<0.5):
                y_hat[i]=0
            else:
                y_hat[i]=1
        
        print("accuracy: " +  str(accuracy_score(list(self.y_test), list(y_hat))))
        print(confusion_matrix(list(self.y_test), list(y_hat)))    
        print("F1 score: " + str(f1_score(self.y_test, y_hat, average='weighted')))

if __name__ == '__main__':
    model_instance = SentimentAnalysisModel()
    model_instance.preprocess()
    model_instance.modelSelection('randomEmbeddings')
    model_instance.splitAndTraining(10, 1)
    model_instance.predict(model_instance.X_test, model_instance.Y_test)