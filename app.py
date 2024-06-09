from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import cv2
import re
import os

app = Flask(__name__)

# Ensure the static/uploads directory exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load crop yield prediction model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Load soil classification model
soil_model = load_model('soil.h5')

# Function to make soil classification prediction
def make_soil_prediction(image_fp):
    img = cv2.imread(image_fp)
    img = cv2.resize(img, (256, 256))  # Resize image to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    class_names = ["Gravel", "Sand", "Silt"]
    prediction = class_names[np.argmax(soil_model.predict(img))]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict_crop_yield", methods=['POST'])
def predict_crop_yield():
    if request.method == 'POST':
        Year = float(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', prediction=prediction, prediction_type="Crop Yield Prediction")

@app.route("/predict_soil_type", methods=['POST'])
def predict_soil_type():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            soil_prediction = make_soil_prediction(filepath)
            return render_template('index.html', soil_prediction=soil_prediction, prediction_type="Soil Type Prediction", filepath=filepath)

if __name__ == "__main__":
    app.run(debug=True)


