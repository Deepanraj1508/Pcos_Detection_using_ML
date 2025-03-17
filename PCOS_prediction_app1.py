import os
import pickle
import joblib
import numpy as np
import logging
from flask import Flask, request, render_template
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to load models safely
def load_model(filename):
    if not os.path.exists(filename):
        logging.error(f"Model file {filename} not found!")
        raise FileNotFoundError(f"Model file {filename} not found!")
    
    try:
        if filename.endswith(".joblib"):
            model = joblib.load(filename)  
        elif filename.endswith(".h5"):
            model = keras.models.load_model(filename)
        else:
            with open(filename, 'rb') as f:
                model = pickle.load(f)  
        
        logging.info(f"Successfully loaded model: {filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {filename}: {str(e)}")
        raise

# Load the models
MODEL_GENERAL_PATH = "model_files/catboost_general_information.joblib"
MODEL_MEDICAL_PATH = "model_files/catboost_general_and_medical_information.joblib"
MODEL_IMAGE_PATH = "model_files/image_binary_transformer_model.h5"

model_general = load_model(MODEL_GENERAL_PATH)
model_medical = load_model(MODEL_MEDICAL_PATH)
model_image = load_model(MODEL_IMAGE_PATH)

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template("menu.html")

@app.route('/general_information')
def general_information():
    return render_template("general_information.html")

@app.route('/general_and_medical_information')
def general_and_medical_information():
    return render_template("general_and_medical_information.html")

@app.route('/image_prediction')
def image_prediction():
    return render_template("image_prediction.html")

# text-based predictions
def make_prediction(model, form_data):
    try:
        int_features = [float(i) for i in form_data.values()]
        array_features = np.array([int_features])

        if array_features.shape[1] != model.n_features_in_:
            logging.error(f"Input feature mismatch! Expected {model.n_features_in_}, got {array_features.shape[1]}.")
            return "Invalid input data: Feature count mismatch."

        prediction = model.predict(array_features)
        return "The patient is likely to have PCOS!" if prediction[0] == 1 else "The patient is not likely to have PCOS!"
    except ValueError as ve:
        logging.error(f"Value error: {str(ve)}")
        return "Invalid input: Please enter valid numbers."
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return "Error in processing the prediction. Please check input values."

# image-based predictions
def predict_image(image_path):
    try:
        #Load and Preprocess Image
        image = load_img(image_path, target_size=(64, 64), color_mode='grayscale')
        img_array = img_to_array(image) / 255.0  # Normalize
        img_array = img_array.reshape(1, 64, 64, 1)  # Reshape for ViT input

        #Make Prediction using ViT model
        prediction = model_image.predict(img_array)
        predicted_class = np.argmax(prediction)  # Get class index

        #Load Class Labels
        class_labels = ["NotInfected", "Infected"] 
        predicted_label = class_labels[predicted_class]

        return f"Prediction: {predicted_label}"
    
    except Exception as e:
        logging.error(f"Image prediction error: {str(e)}")
        return "Error processing image. Please try again."

@app.route('/predict', methods=['POST'])
def predict():
    result_text = make_prediction(model_general, request.form)
    return render_template('general_information.html', result=result_text)

@app.route('/pred', methods=['POST'])
def pred():
    result_text = make_prediction(model_medical, request.form)
    return render_template('general_and_medical_information.html', result=result_text)

@app.route('/predict_image', methods=['POST'])
def predict_image_route():
    if 'file' not in request.files:
        return render_template('image_prediction.html', result="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('image_prediction.html', result="No selected file.")
    
    file_path = os.path.join("static/uploads", file.filename)
    file.save(file_path)
    
    result_text = predict_image(file_path)
    return render_template('image_prediction.html', result=result_text, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
