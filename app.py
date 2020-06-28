from flask import Flask, request, url_for, redirect, render_template
import pickle

import numpy as np

app = Flask(__name__)

model = pickle.load(open('./churn-model-85.pkl','rb'))
 @app.route('/')

 def hello_world():
     return render_template('./churn-modeling.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.from.values()]
    x = features[5]
    if x=='YES':
        x = 1
    else:
        x = 0
    features[5]=x
    x = features[6]
    if x=='YES':
        x = 1
    else:
        x = 0
    features[6]=x

    x = features[8]
    if x=='M':
        x = 1
    else:
        x = 0
    features[8]=x