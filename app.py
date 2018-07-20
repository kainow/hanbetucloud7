import os
from flask import Flask, request, redirect, url_for 
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.models import Sequential, load_model
import keras,sys
import numpy as np 
from PIL import Image
from flask import send_from_directory


classes = ["altocumulus","stratus","nimbostratus","altostratus","cirrocumulus","cirrostratus","cirrus","cumulonimbus","cumulus","stratocumulus"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENTIONS = set(['png','jpg','gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('ファイルがないです')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('ファイルがないです')
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

            return "その雲はおそらく" +classes[predicted] +",で、その確率は" + str(percentage) + "%ぐらいです！"


def main():
    return flask.render_template("index.html")


            #return redirect(url_for('uploaded_file',filename=filename))
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>雲の画像をアップロードしてください</title></head>
    <body>
    <h1>雲の画像をアップロードしてください！どういった種類の雲か判別します！</h1>
    <form method =post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''




@app.route('/uploads/<filename>')

def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
if __name__ == '__main__':
    app.run()


