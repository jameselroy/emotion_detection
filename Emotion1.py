#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import the rquired libraries.
import numpy as np
import cv2
import os
from keras.models import load_model
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
#classifier =load_model('model_78.h5')
classifier =load_model('C:/Users/James/anaconda2/model/model_78.h5')

# load weights into new model
#classifier.load_weights("model_weights_78.h5")
classifier.load_weights("C:/Users/James/anaconda2/model/model_weights_78.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Face Emotion Detection Application ")
    activiteis = ["Home", "Live Face Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by JAMES BARNES
            """)

    # Homepage.
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#02b2fc;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            Start Your Real Time Face Emotion Detection.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
        * Emotions are a powerful and extraordinary part of being human. 
        * Emotional expression through body language, such as various facial expressions helps people to understand what actually your feeling at that moment of time.
        * So this emotion detection application will try to understand indivual by taking into consideration their facial emotions.
        
        steps to follow
        1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        2. This will redirect to emotion detection page .
                 """)

    # Live Face Emotion Detection.
    elif choice == "Live Face Emotion Detection":
        st.header("Welcome to Emotion Detection !!!")
        
        st.write("1. Kindly click on  Start button to open your camera and give permission for prediction")
        st.write("2. provide  permission for prediction.")
        st.write("3. Look into the screen to  predict your emotion.")
        st.write("4. Click stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # About.
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achived through openCV.
                                    </h4>
                                    </div>
                                    </br>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)


    else:
        pass


if __name__ == "__main__":
    main()


# In[ ]:




