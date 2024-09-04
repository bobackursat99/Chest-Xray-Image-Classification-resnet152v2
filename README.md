# Classificaiton of Chest X-ray Images with Resnet152v2

This is my senior project classifies chest X-ray images into four classes: COVID-19, pneumonia, lung opacity, and normal using a deep learning model (ResNet152v2) with transfer learning.

## Features
- **Deep Learning Model**: Utilizes ResNet152v2 architecture.
- **Flask Web Interface**: User-friendly interface for uploading and classifying X-ray images.
- **Custom Model Training**: Includes a Jupyter notebook for training the model with your own dataset.

## Installation

1. **Clone the repository**:
    ```git clone https://github.com/your-username/Chest-Xray-Image-Classification-resnet152v2.git
       cd Chest-Xray-Image-Classification-resnet152v2
    ```

2. **Install dependencies**:
    ```bash
       pip install -r requirements.txt
    ```
## Model Training

### Prepare Your Dataset:

Organize your chest X-ray images into four folders named you wanted classify classes.
Place these folders inside a main directory, e.g., dataset/.

### Train the Model:

Open the train_model.ipynb Jupyter notebook.
Update the dataset path in the notebook to point to your custom dataset.
Run the cells in the notebook to start training the model using ResNet152v2 architecture.
Save the Trained Model:

After training, the model will be saved as an .h5 file (e.g., model.h5) in the project directory.

4. **Run the application**:
    ```bash
    python app.py
    ```

## Usage
Upload an X-ray image through the web interface to get the classification results.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
