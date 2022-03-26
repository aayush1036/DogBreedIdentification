import pickle
from flask import Flask,render_template,url_for,request
from utils import * 
from tensorflow.keras.models import load_model
import json 
import mysql.connector

with open('config.json', 'rb') as f:
    config = json.load(f)

UPLOAD_PATH = 'static/uploads'
MODEL_PATH = 'best_InceptionResnetV2.h5'
if not checkUploadPath(UPLOAD_PATH):
    createUploadPath(UPLOAD_PATH)

model = load_model(MODEL_PATH)

with open('labelDict.pkl', 'rb') as f:
    predictionDict = pickle.load(f)

allowedFileTypes = ['jpg','png','jpeg']
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/GetData', methods=['GET'])
def getData():
    return render_template('getData.html')

@app.route('/GetData',methods=['POST'])
def predict():
    if request.method == 'POST':
        username = request.form['username']
        address = request.form['address']

        if request.files:
            image = request.files['image']
            if image.filename == '':
                statusCode = 'NoFile'
                return render_template('getData.html',statusCode=statusCode, pred=None)
            if image.filename.rsplit('.',1)[1] not in allowedFileTypes:
                statusCode = 'InvalidFileType'
                return render_template('getData.html',statusCode=statusCode, pred=None)
            else:
                statusCode = 'Success'
                imgPath = os.path.join(UPLOAD_PATH,image.filename) 
                image.save(imgPath)
                pred,conf = predictNew(model=model, path=imgPath, labelDict=predictionDict)
                message = f'The model predicted {pred} with {conf:.3%} confidence'
                con = mysql.connector.connect(
                    host = config.get('host'),
                    port = config.get('port'),
                    username = config.get('username'),
                    password = config.get('password'),
                    auth_plugin = config.get('auth_plugin'),
                    database = config.get('database')
                )
                cursor = con.cursor()
                query = f"INSERT INTO ENTRIES (USERNAME, ADDRESS, BREED) VALUES ('{username}', '{address}', '{pred}');"
                cursor.execute(query)
                con.commit()
                return render_template('getData.html',statusCode=statusCode,message=message,conf=conf)
        
if __name__ == '__main__':
    app.run(debug=True)