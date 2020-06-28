from flask import Flask, request, url_for, redirect, render_template
import pickle

import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

model = pickle.load(open('./churn-model-85','rb'))
@app.route('/')

def hello_world():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    features = [int(x) for x in request.form.values()]
    x = features[1]
    if x=='SPAIN':
        x = 2
    elif x=='FRANCE':
        x = 0
    else:
        x = 1
    features[1]=x
    
    x = features[7]
    if x=='YES':
        x = 1
    else:
        x = 0
    features[7]=x

    x = features[8]
    if x=='YES':
        x = 1
    else:
        x = 0
    features[8]=x
    print(features)
    final = [np.array(features)]
    #print(type(final))
    print(final)
    #print(final.shape)
    pred = model.predict(final)
    if pred >= 0.5:
        output = "True"
    else:
        output = "False"

    if output == "False":
        return render_template('home.html', pred='Your Customer might Stop using your bank!')
    else:
        return render_template('home.html', pred='Your Customer ain\'t stop using your bank!')

if __name__ == '__main__':
    app.run(debug=True)