import io
import base64
from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load YOLO lung detection model
model = YOLO("best_harshita.pt")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            # Read image into memory
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Predict with YOLO
            results = model.predict(img, imgsz=640)

            # Draw bounding boxes
            result_img = results[0].plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(result_img_rgb)

            # Convert to base64
            img_io = io.BytesIO()
            pil_img.save(img_io, 'PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Process detections
            detections = results[0].boxes
            message = "Detections Found!" if len(detections) > 0 else "No Detections Found"

            detections_data = []
            lung_count, abnormal_count, other_count = 0, 0, 0

            lung_keywords = ["lung", "left_lung", "right_lung"]
            abnormal_keywords = ["opacity", "nodule", "tb", "pneumonia"]

            for i, box in enumerate(detections, start=1):
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                label = results[0].names.get(cls_id, "Unknown")

                if any(k in label.lower() for k in lung_keywords):
                    lung_count += 1
                    detections_data.append({
                        "type": "Lung Region",
                        "label": label,
                        "confidence": f"{confidence:.2f}",
                        "icon": "🫁",
                        "color": "primary"
                    })
                elif any(k in label.lower() for k in abnormal_keywords):
                    abnormal_count += 1
                    detections_data.append({
                        "type": "Abnormality",
                        "label": label,
                        "confidence": f"{confidence:.2f}",
                        "icon": "⚠",
                        "color": "danger"
                    })
                else:
                    other_count += 1
                    detections_data.append({
                        "type": "Other Detection",
                        "label": label,
                        "confidence": f"{confidence:.2f}",
                        "icon": "🔎",
                        "color": "secondary"
                    })

            summary_counts = {
                "lungs": lung_count,
                "abnormalities": abnormal_count,
                "others": other_count
            }

            return render_template(
                'result.html',
                message=message,
                image_data=img_base64,
                detections=detections_data,
                summary_counts=summary_counts
            )

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)














