from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
model = load_model("downsyndromemodel.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))
    img = img / 255.0
    return img.reshape(1, 250, 250, 3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Save the file to process
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_image.jpg")
    file.save(file_path)

    # Preprocess the image
    processed_image = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(processed_image)
    pred_class = np.round(prediction).astype(int)

    # You can adjust the response based on your requirement
    response = {
        'prediction': int(pred_class[0, 0]),
        'class_name': 'down' if pred_class[0, 0] == 1 else 'healthy'
    }

    return jsonify(response)

# Define the folder where uploaded images will be stored
app.config['UPLOAD_FOLDER'] = 'uploads'

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
