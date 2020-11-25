import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('prediction_model.sav', 'rb'))

@app.route('/')
def home() -> 'html':
    return render_template('index.html')


@app.route('/strength', methods = ['POST'])
def predict():

    cement = float(request.form['cement'])
    slag = float(request.form['slag'])
    ash = float(request.form['ash'])
    water = float(request.form['water'])
    plastic = float(request.form['plastic'])
    coarse = float(request.form['coarse'])
    fine = float(request.form['fine'])
    age = float(request.form['age'])

    X = [[cement], [slag], [ash], [water], [plastic], [coarse], [fine], [age]]
    X = np.transpose(X)

    prediction = model.predict(X)

    return render_template('index.html', prediction = prediction)


if __name__ == '__main__':
    app.run()
