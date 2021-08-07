import uvicorn
from fastapi import FastAPI
from texts import text
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = FastAPI()
pickle_in_1 = open("classifier.pkl","rb")
pickle_in_2 = open("vectorizer.pkl", "rb")
classifier=pickle.load(pickle_in_1)
tfidfVectorizer = pickle.load(pickle_in_2)

@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To EntHire Airline Sentiment Classification': f'{name}'}


@app.post('/predict')
def predict_sentiment(data:text):
    data = data.dict()
    review=data['review']
    
    sentences = [review]
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
       
    final = []
    for word in words_list:
        subfinal = ' '.join([words for words in word])
        final.append(subfinal)
    
    X = tfidfVectorizer.transform(final).toarray()
    
    prediction = classifier.predict(X)
    
    if(prediction[0]>0.5):
        prediction="Positive"
    else:
        prediction="Negative"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload