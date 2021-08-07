# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 19:46:08 2021

@author: HP
"""


import pandas as pd
import re
import nltk
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import *
from keras import regularizers
import matplotlib.pyplot as plt
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

df['l'] = df['clean_text'].apply(lambda x: len(str(x).split(' ')))
sequence_length = df.l.max()
max_features = 20000 # this is the number of words we care about

tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
tokenizer.fit_on_texts(df['clean_text'].values)

# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(df['clean_text'].values)

# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)

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
    embedding_dim = 200 # Kim uses 300 here
    num_filters = 100

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
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    flatten = Flatten()(concatenated_tensor)
    
    # do dropout and predict
    dropout = Dropout(0.5)(flatten)
    output = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    batch_size = 64 # Kim uses 50 here, I have a slightly smaller sample size than num
    history = model.fit(x_train,y_train, epochs=15, batch_size=batch_size, verbose=1, validation_data = (x_test, y_test), shuffle=True)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    y_hat = model.predict(x_test)
    
    for i in range(0, len(y_hat)):
        if(y_hat[i]<0.5):
            y_hat[i]=0
        else:
            y_hat[i]=1
            
    print("accuracy:" +  str(accuracy_score(list(y_test), list(y_hat))))

    print(confusion_matrix(list(y_test), list(y_hat)))
    
    print("F1 score: " + str(f1_score(y_test, y_hat, average='weighted')))

