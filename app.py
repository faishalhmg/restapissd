import tensorflow as tf
from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

# Loading the saved_model
PATH_TO_SAVED_MODEL = "saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# Loading the label_map
label_map = {
    1: "akarPatahmati",
    2: "batangakarpatah",
    3: "batangpecah",
    4: "brumakarbatang",
    5: "cabangpatahmati",
    6: "daunberubahwarna",
    7: "daunpucuktunasrusak",
    8: "gerowong",
    9: "hilangpucukdominan",
    10: "kanker",
    11: "konk",
    12: "liana",
    13: "lukaterbuka",
    14: "percabanganbrumberlebihan",
    15: "resinosisgumosis",
    16: "sarangrayap"
}

class_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (128, 0, 0),    # Dark Red
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Blue
    (128, 128, 0),  # Dark Cyan
    (0, 128, 128),  # Dark Yellow
    (128, 0, 128),  # Dark Magenta
    (255, 128, 0),  # Orange
    (128, 255, 0),  # Lime
    (0, 128, 255),  # Sky Blue
    (255, 0, 128)   # Pink
]

def detect_objects(frame):
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy().tolist() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    image_np_with_detections = image_np.copy()

    # Draw bounding boxes on image
    for box, score, class_id in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_classes']):
        if score > 0.5:
            height, width, _ = image_np_with_detections.shape
            ymin, xmin, ymax, xmax = box
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)
            offset = 40  # Jumlah offset yang ingin Anda gunakan
            class_name = label_map[int(class_id)]

            # Get class color based on class_id
            class_color = class_colors[int(class_id) % len(class_colors)]

            # Draw bounding box
            cv2.rectangle(image_np_with_detections, (left, top + offset), (right, bottom), class_color, 2)

            # Add background box for label
            cv2.rectangle(image_np_with_detections, (left, top + offset - 20), (right, top + offset), class_color, -1)

            # Add class label text
            cv2.putText(image_np_with_detections, f"{class_name} ({round(score * 100, 2)}%)", (left, top + offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return image_np_with_detections

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output_frame = detect_objects(frame)

        ret, buffer = cv2.imencode('.jpg', output_frame)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/detect_image', methods=['POST'])
def detect_image():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_img = detect_objects(img)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', output_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
