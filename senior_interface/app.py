from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set the path where uploaded files will be stored
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the AI model
model = load_model(r"C:\Users\Asus Hn004W\Desktop\pnomoni_model_Senior_sunum_real.h5")

# Define class names
class_names = ['Covid', 'Lung Opacity', 'Normal', 'PNEUMONIA']

# Global list to store recent predictions
recent_predictions = []

# Function to preprocess the uploaded image
def preprocess_image(image):
    img_width = 224
    img_height = 224
    image = image.resize((img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Function to make predictions using the loaded model
def predict_image(image):
    return model.predict(image)

# Home route
@app.route('/')
def home():
    return render_template('index.html', recent_predictions=recent_predictions)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file part', recent_predictions=recent_predictions)

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return render_template('index.html', error='No selected file', recent_predictions=recent_predictions)

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and preprocess the uploaded image
        uploaded_image = load_img(filepath)
        processed_image = preprocess_image(uploaded_image)

        # Make prediction using the loaded model
        predictions = predict_image(processed_image)

        # Get the predicted class label
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Update recent predictions list
        recent_predictions.append((url_for('static', filename='uploads/' + filename), predicted_class))
        if len(recent_predictions) > 5:
            recent_predictions.pop(0)

        # Pass the predicted class and file path to the template
        return render_template('index.html', prediction=predicted_class, image_url=url_for('static', filename='uploads/' + filename), recent_predictions=recent_predictions)

    # If the file extension is not allowed
    return render_template('index.html', error='File extension not allowed', recent_predictions=recent_predictions)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
