from flask import Flask, render_template, request,redirect,flash,url_for
from PIL import Image 
import pickle 
import os
import numpy as np
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

with open('model2.pkl', 'rb') as file:  
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def home():
    pixels,full_filename,exist = upload_save_file()
    if exist:
        if request.form.get('predict'):
            prediction = model.predict(np.array([pixels]))
            #score = accuracy_score(pixcel.flatten(),prediction)
            classification = str(prediction)
            return render_template('home.html',user_image = full_filename,prediction=classification, filename=full_filename)
        else:
            return  index() 
    else:
        return  index()

def upload_save_file():
    uploaded_file = request.files['file'] 
    if uploaded_file.filename != "":
        print("upload file",uploaded_file.filename) 
        try:
            img = plt.imread(uploaded_file) 
            im =img/255.0
            print("before reshape",im.shape)
            pixcel = im.reshape(im.shape[0]*im.shape[1])
            print("after reshape",pixcel.shape)
            flash('Image successfully uploaded and displayed below')
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename) 
            print("full filename",full_filename)
            plt.imsave(full_filename,img)
            print("uploaded file after:",full_filename)
            return pixcel,full_filename, True
        except:
            return None,None, False
    return None,None, False

if __name__ == '__main__':
   app.run(debug = True,port=8080)
