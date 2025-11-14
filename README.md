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





Installation and Setup

1. Clone the repository:

 	git clone <repository\_url>

 	cd Malaria-Detection-Kit



2\. Install dependencies:

 	pip install -r requirements.txt



3\. Download and prepare the dataset (Malaria-Dataset folder) in the project root.



4\. Train the model or use pre-trained model in models/model.h5:

 	python malaria.py



5\. Run the Flask app:

 	python app.py



6\. Open a browser and go to http://127.0.0.1:5000 to interact with the web UI.



7\. Upload cell images:

 	Use the form to upload an image and view predictions.



Malaria-Detection-Kit guide:

Malaria-Detection-Kit guide.pdf provides EDA, data visualization, and step-by-step modeling.





Model Details

CNN with three convolutional blocks, BatchNorm, Dropout, Dense layers.

Input: 50x50x3 RGB cell images.

Output: Classification as "Malaria Parasitized" or "Normal".

Best accuracy: (your result, e.g. 90%).





Known Limitations

Deploy as a research/teaching tool—not for clinical use.

Image quality and model generalizability depend on dataset used.



Credits \& References

Kaggle dataset: Cell Images for Detecting Malaria

Keras, Flask, and other open-source libraries.



