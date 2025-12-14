# Malaria Detection Using Machine Learning
Overview
This project uses a convolutional neural network (CNN) to detect malaria parasites in microscope cell images. It includes image preprocessing, model training, and a Flask-powered web app for live predictions.
Dataset

Source: Cell Images for Detecting Malaria (Kaggle)

Data contains labeled microscopic cell images: "Parasitized" and "Uninfected".

Features

CNN for malaria cell classification (Keras/TensorFlow).

Training and evaluation with balanced class counts.

Flask web app with image upload and prediction.

Visuals for training/validation curves and results.

Directory Structure:
```bash
 	Malaria Detection Kit/

 	│

 	├── Malaria-Dataset/          # Blood cell images

 	├── models/                   # Saved Keras model (.h5)

 	├── output/                   # Generated plots and output

 	├── static/                   # CSS, JS, images (e.g., logo.png)

 	├── templates/                # Flask HTML templates (base.html, index.html)

 	├── uploads/                  # Temp uploaded images during inference

 	├── app.py                    # Flask web app

 	├── malaria.py                # Training pipeline and model definition

 	├── requirements.txt          # Required Python packages

 	└── README.md                 # Project overview and instructions
```

Installation and Setup

Clone the repository:
```bash
git clone https://github.com/SyedRaheeb-py/Malaria-Detection-Kit.git
cd Malaria-Detection-Kit
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Download and prepare the dataset (Malaria-Dataset folder) in the project root.

Train the model or use pre-trained model in models/model.h5:
```bash
python malaria.py
```
Run the Flask app:
```bash
python app.py
```
Open a browser and go to http://127.0.0.1:5000 to interact with the web UI.
Upload cell images:
Use the form to upload an image and view predictions.

Model Details

CNN with three convolutional blocks, BatchNorm, Dropout, Dense layers.

Input: 50x50x3 RGB cell images.

Output: Classification as "Malaria Parasitized" or "Normal".

Credits & References

Kaggle dataset: Cell Images for Detecting Malaria

Keras, Flask, and other open-source libraries.



