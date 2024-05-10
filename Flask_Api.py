from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__)

# Load your trained model
model = load_model("downsyndromemodel.h5")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))
    img = img / 255.0
    return img.reshape(1, 250, 250, 3)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    # Save the file to process
    file_path = "temp_image.jpg"
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

if __name__ == '__main__':
    app.run(debug=True)
