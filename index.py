# load Flask 
import flask
# from tensorflow import keras
import cv2 as cv2
import numpy as np
import pandas as pd
from flask import url_for
import os
app = flask.Flask(__name__)
# define a predict function as an endpoint 
print(__name__)

# @app.route("/predict", methods=["GET","POST"])
# def predict():
#     data = {"success": False}
#     return 'bgf'
#     # get the request parameters
#     params = flask.request.json
   
#     if (params == None):
#         params = flask.request.args
#     # if parameters are found, echo the msg parameter 
#     if (params != None):
#         data["response"] = params.get("msg")
#         data["success"] = True
#     # return a response in json format 
    
#     return 'flask.jsonify(data)'
import keras


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, 'static/model_config.json')
with open(my_file) as json_file:
    json_config = json_file.read()
model = keras.models.model_from_json(json_config)
my_file = os.path.join(THIS_FOLDER, 'static/weights_only.h5')

# Load weights
model.load_weights(my_file)
@app.route('/upload', methods=['POST'])
def upload():
    try:
            #read image file string data
        filestr = flask.request.files['image'].read()
        #convert string data to numpy array
        file_bytes = np.frombuffer(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_REDUCED_COLOR_2)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       
      
        img_size=(300, 300)
       
        img=cv2.resize(img, img_size)    
        img=np.expand_dims(img, axis=0)
        p= np.squeeze (model.predict(img,use_multiprocessing=False,workers=1))
        index=np.argmax(p)  
        my_file = os.path.join(THIS_FOLDER, 'static/class_dict (1).csv')
        class_df=pd.read_csv(my_file)          
        prob=p[index]
        classname=class_df['class'].iloc[index]
        print(classname,prob)
        return flask.jsonify(classname=classname,prob=str(prob))
    except Exception as err:
        print('error',err)
# start the flask app, allow remote connections
app.run(host='0.0.0.0')