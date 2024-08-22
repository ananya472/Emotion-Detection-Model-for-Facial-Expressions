import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# Load the trained model
model = load_model(r'C:\Users\anany\Desktop\project\final_model.h5')

# Define the input shape and label encoder
input_shape = (48, 48, 1)  # Adjust this if your model has a different input shape
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Adjust labels based on your dataset

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(48, 48), color_mode="grayscale")  # Adjust target size and color mode if needed
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize the image
    return image

@app.route('/')
def index():
    return render_template('img-index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the file temporarily
    file_path = f"temp/{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    
    # Preprocess the image
    image = preprocess_image(file_path)
    
    # Predict the sentiment
    prediction = model.predict(image)
    predicted_class = labels[np.argmax(prediction)]
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
