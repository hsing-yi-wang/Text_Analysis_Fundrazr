from flask import Flask,render_template,url_for,request
import pickle
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import SVC

with open(f'model/fundrazr.clf2.pkl', 'rb') as f:
    model = pickle.load(f)

with open(f'model/vect.pkl', 'rb') as f1:
    vect = pickle.load(f1)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')



@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = str(message)
        tokenizer = RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(data)
        words_lower = [x.lower() for x in words]
        stop_words = set(stopwords.words('english'))
        valued_words = [w for w in words_lower if not w in stop_words]
        valued_words = [' '.join(valued_words)]
        test = vect.transform(valued_words).astype('float')
        my_prediction = model.predict(test)
    return render_template('result.html', prediction=my_prediction)



if __name__ == '__main__':
    app.run(debug=True)