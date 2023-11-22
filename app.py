import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import time
import webbrowser
from streamlit_webrtc import webrtc_streamer

def detect_and_predict_mask(frame, faceNet, maskNet, threshold):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            locs.append((startX, startY, endX, endY))
            preds.append(maskNet.predict(face)[0].tolist())

    return locs, preds
# SETTINGS
MASK_MODEL_PATH = "model/emotion_model.h5"
FACE_MODEL_PATH = "face_detector"
THRESHOLD = 0.5

# Load Models
faceNet = cv2.dnn.readNet("face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model(MASK_MODEL_PATH)

labels = ["happy", "neutral", "sad"]

st.title("Facial Emotion Detection and recommendation ")
# Initialize emotion_result outside the if block
emotion_result = "happy"
# Function to read video stream and perform detection
def detect_emotion():
    global emotion_result  # Use the global keyword to modify the global variable

    vs = cv2.VideoCapture(0)

    start_time = time.time()

    while time.time() - start_time < 2:
        ret, frame = vs.read()
        frame = cv2.resize(frame, (400, 300))
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Detect faces and emotions
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, THRESHOLD)

        # Loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # Unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            label = str(labels[np.argmax(pred)])

            if label == "happy":
                emotion_result = "Happy"
            elif label == "neutral":
                emotion_result = "Neutral"
            elif label == "sad":
                emotion_result = "Sad"

            cv2.putText(original_frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                        2)
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), (255, 255, 255), 2)

        # Display the output frame in Streamlit
        st.image(original_frame, channels="BGR", use_column_width=True)

    # Stop the video stream
    vs.release()

    return None

lang = st.text_input("Language")
hero = st.text_input("Singer")

# Start the video stream and capture emotion
if st.button("Capture Emotion"):
    emotion_result = detect_emotion()
    #st.write(f"Detected Emotion: {emotion_result}")

    # Open YouTube with the appropriate search query based on emotion, language, and hero
    youtube_url = f"https://www.youtube.com/results?search_query={emotion_result}+{lang}+{hero}+songs"
    st.write("Redirecting to YouTube...")
    webbrowser.open(youtube_url)