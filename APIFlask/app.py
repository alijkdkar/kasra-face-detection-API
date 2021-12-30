
from flask import Flask , request, jsonify
import base64
import io
from PIL import Image
from flask_cors import CORS
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model


from datetime import datetime
import re


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Hello, Flask!"



@app.route("/hello/<image>",methods=["POST"])
def hello_there(image):
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)

    return jsonify({'msg': 'success', 'size': [img.width, img.height]})





@app.route("/FaceDet//<path:url>",methods=["POST"])
def FaceDetection(url):
    step=0
    try:
        # Base64 DATA
        if "data:image/jpeg;base64," in url:
            base_string = url.replace("data:image/jpeg;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(io.BytesIO(decoded_img)).convert('L')
            img = img.resize((150,100))
            step=1
            data = np.asarray( img, dtype="int32" )

            # file_name = file_name_for_base64_data + ".jpg"
            # img.save(file_name, "jpeg")

        # Base64 DATA
        elif "data:image/png;base64," in url:
            base_string = url.replace("data:image/png;base64,", "")
            decoded_img = base64.b64decode(base_string)
            img = Image.open(io.BytesIO(decoded_img)).convert('L')
            img=img.resize((150,100))
            data = np.asarray( img, dtype="int32" )
            step=2
            
            # file_name = "file_name_for_base64_data" + ".png"
            # img.save(file_name, "png")

        # Regular URL Form DATA
        else:
            response = request.get(url)
            img = Image.open(io.BytesIO(response.content)).convert("L")
            step=3
            img = img.resize((150,100))
            data = np.asarray( img, dtype="int32" )
            # file_name = file_name_for_regular_data + ".jpg"
            # img.save(file_name, "jpeg")
        
    # ----- SECTION 3 -----    
        status =  jsonify({'msg': 'success'+str(step), 'size': data.shape})
    
        # model = load_model('mnistModelLearned.h5')
        # image_array = np.array(img)
        # image_array = image_array.astype("float32") / 255
        # image_array = image_array[:,:,0]
        # imag = image_array.reshape(1,28,28,1)
        # target = model.predict(imag)
        
        # iin =target.argmax()
        # rrr = str(iin)

    except Exception as e:
        status = "Error! = " + str(e)

        
    return status 





if __name__ == "__main__":
    app.run(debug=True)
    