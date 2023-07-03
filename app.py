import tensorflow as tf
import cv2
import numpy as np
import streamlit as st

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

# Loading the saved_model
PATH_TO_SAVED_MODEL = "saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

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
            offset = 10  # Jumlah offset yang ingin Anda gunakan
            class_name = label_map[int(class_id)]

            # Get class color based on class_id
            class_color = class_colors[int(class_id) % len(class_colors)]

            # Draw bounding box
            cv2.rectangle(image_np_with_detections, (left, top + offset), (right, bottom), class_color, 2)

            # Add background box for label
            cv2.rectangle(image_np_with_detections, (left, top + offset - 20), (right, top + offset), class_color, -1)

            # Add class label text
            cv2.putText(image_np_with_detections, f"{class_name} ({round(score * 100, 2)}%)", (left, top + offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_np_with_detections

def main():
    st.title("Object Detection")

    option = st.sidebar.selectbox("Choose an option", ["Image", "Camera"])

    if option == "Image":
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if image_file is not None:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            output_img = detect_objects(img)
            st.image(output_img, channels="BGR")

    elif option == "Camera":
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            output_frame = detect_objects(frame)
            st.image(output_frame, channels="BGR")

    video_capture.release()

if __name__ == '__main__':
    main()
