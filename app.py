import pickle
from flask import Flask, jsonify, request, url_for,render_template
import numpy as np
import pandas as pd
import json



app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=np.array(list(data.values())).reshape(1,-1)
    
    final_input=scaler.transform (new_data)
    output=regmodel.predict(final_input)
    print(output[0])
    return jsonify(output[0])



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform (np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template('home.html',prediction_text="The predicted House  price is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)


