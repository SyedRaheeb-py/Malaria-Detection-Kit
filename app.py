# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 19:20:27 2022

@author: SyedRaheeb
"""

from __future__ import division, print_function
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = 'models/model.h5'

#Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Start serving...')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(50,50))
    img = image.img_to_array(img)
    img = img / 255.0
    print("INFERENCE:", img.dtype, img.min(), img.max())
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    print("Prediction probabilities: ", preds)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

os.makedirs('uploads', exist_ok=True)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pred = model_predict(file_path, model)
        os.remove(file_path)
        str1 = 'Malaria Parasitized'
        str2 = 'Normal'
        if pred[0] == 1:
            return str1
        else:
            return str2
    return None

if __name__ == '__main__':
        app.run()