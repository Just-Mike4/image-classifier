from flask import Flask, request, jsonify, render_template 
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
app = Flask(__name__)
dic={0:'cat', 1:'dog', 2:'Male', 3:'Female'}
model= load_model("model8.h5")
model.make_predict_function()
def predict_label(img_path):
          i=load_img(img_path,target_size=(64,64))
          i=img_to_array(i)
          i=i.reshape(1,64,64,3)
          p=model.predict(i)
          Pred=np.argmax(p,axis=1)
          return dic[Pred[0]]

@app.route("/", methods=['GET','POST'])
def home():
          return render_template("image_html.html")


@app.route("/submit",methods=["GET","POST"])
def predict():
          if request.method == 'POST':
                    img=request.files['imagefile']
                    img_path='images'+ img.filename
                    img.save(img_path)
                    p=predict_label(img_path)
          return render_template("image_html.html", prediction = p, img_path=img_path)

if __name__=="__main__":
          app.run(debug=True)
