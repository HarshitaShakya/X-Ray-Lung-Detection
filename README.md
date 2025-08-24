# X-Ray-Lung-Detection

A web-based AI application that detects lung regions and abnormalities from X-ray images using YOLOv8 and Flask. Users can upload chest X-ray images and receive a detailed detection report with bounding boxes, confidence scores, and a summary of findings.

# Features

-Detect lungs, abnormalities, and other objects in X-ray images
-Visualize predictions with bounding boxes
-Summarized medical analysis report
-Responsive and clean web UI using Bootstrap
-Ready to run locally with Flask
-Getting Started

# Install Dependencies
pip install -r requirements.txt


# Required Python packages:
flask
ultralytics 
opencv-python
numpy
pillow
torch
torchvision

# Run the App Locally
python app.py


Open your browser at http://127.0.0.1:5000
 and upload a chest X-ray image for analysis.

# Project Structure
lung-xray-detection/
│
├─ app.py                   # Main Flask app
├─ best_harshita.pt         # Trained YOLOv8 model
├─ templates/               # HTML templates
│   ├─ index.html           # Upload page
│   └─ result.html          # Detection results page
├─ static/                  # Optional: CSS/JS/images
├─ requirements.txt         # Python dependencies
└─ README.md                # Project documentation

# Usage
-Open the app in a browser.
-Upload a chest X-ray image (.jpg, .png, .jpeg).
-Wait for YOLOv8 to process the image.
-View detection results:
-Image with bounding boxes
-Summary of detected lungs, abnormalities, and other findings
-Detailed table with confidence scores


Can be deployed on any server supporting Python and Flask (e.g., Heroku, AWS, or local server).

Make sure to include the YOLOv8 model file (best_harshita.pt).
