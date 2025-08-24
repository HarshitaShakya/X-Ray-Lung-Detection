<img width="1883" height="894" alt="Screenshot 2025-08-23 153649" src="https://github.com/user-attachments/assets/07195e5b-b2a1-4a30-9a0b-b6b9b846196b" /><img width="1916" height="1018" alt="Screenshot 2025-08-23 154524" src="https://github.com/user-attachments/assets/1b7aa03b-d7b3-4e7f-830e-0b4e7b1ad768" /># X-Ray-Lung-Detection

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



<img width="1919" height="1008" alt="Screenshot 2025-08-23 154552" src="https://github.com/user-attachments/assets/294f1a11-292e-4e48-87bd-f27486cc3c72" />





