import tensorflow as tf
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
app = Flask(__name__)

# Loading the saved_model
PATH_TO_SAVED_MODEL = "saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# Loading the label_map
category_index = label_map_util.create_category_index_from_labelmap("saved_model/pohon_label_map.pbtxt", use_display_name=True)

def detect_objects(frame):
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy().tolist() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        np.array(detections['detection_boxes']),
        np.array(detections['detection_classes']).astype(np.int64),
        np.array(detections['detection_scores']),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.4,
        agnostic_mode=False
    )

    return image_np_with_detections

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = detect_objects(frame)
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

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
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_img = detect_objects(img)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', output_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()