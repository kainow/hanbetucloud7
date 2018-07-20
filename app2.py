# -*- coding: utf-8 -*-
import os
import flask
from flask import Flask, request, redirect, url_for , render_template
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.models import Sequential, load_model
from keras.preprocessing import image
import keras,sys
import numpy as np 
from PIL import Image
from flask import send_from_directory
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import json
import re
import glob



app = flask.Flask(__name__)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))

    x = image.img_to_array(img)

    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)

    return preds

@app.route("/", methods=['GET'])
def main():
    return render_template('index.html')



@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('ファイルがアプロードされていません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('ファイルがアプロードされていません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)

            model = load_model('models/cloudcnn.h5')
            model = model_from_json(open('models/cloudmodel.json').read())

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size,image_size))
            data = np.asarray(image)
            X =[]
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted]*100)

            return flash("その雲はおそらく" +classes[predicted] +",で、その確率は" + str(percentage) + "%ぐらいです！")

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
