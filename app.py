# Importing Necessary Libraries
import os
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template

# Creating Name of Flask Application
imagePredictor = Flask(__name__)


def make_prediction(path, model):
    """This method will perform the preprocessing part for all the incoming images. First it will fetch the uploaded image into
    desired input shape which is fit to the model and then convert it into vector array and then do the prediction part."""
    im = image.load_img(path, target_size=(model.input_shape[1], model.input_shape[2]))
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds


# Displaying the Home Page
@imagePredictor.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# Ajax call triggered from UI to backend to fetch the class in which the object of the image belongs to.
@imagePredictor.route('/predict', methods=['GET', 'POST'])
def upload_and_decode():
    if request.method == 'POST':
        # Retrieving parameters sent from UI side
        file = request.files['file']
        model_name = str(request.form['model_name'])

        # Loading of weights and Building of Model
        model = load_model('TL_models/' + model_name + ".h5")
        model.make_predict_function()

        # Need to save the uploaded image file temporarily before moving ahead with preprocessing
        curr_path = os.path.dirname(__file__)
        file_path = os.path.join(curr_path, secure_filename(file.filename))
        file.save(file_path)

        # Prediction of class of the image
        preds = make_prediction(file_path, model)
        pred_class = decode_predictions(preds=preds, top=1)
        res = str(pred_class[0][0][1])

        # Finally removing the temporary file from location and returning the class name to UI page
        os.remove(os.path.join(curr_path, file.filename))
        return res


if __name__ == '__main__':
    imagePredictor.run(debug=True)
