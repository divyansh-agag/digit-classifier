import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('mnist.h5')

st.title("Digit Classifier (MNIST) with Streamlit")

st.write("Draw a digit (0-9) below:")

# Create canvas
canvas_result = st_canvas(
    fill_color="black", 
    stroke_width=15, 
    stroke_color="white", 
    background_color="black", 
    width=280, height=280, 
    drawing_mode="freedraw", 
    key="canvas",
)
evl=st.button("Predict")
if evl:
    if canvas_result.image_data:
        img = canvas_result.image_data[:, :, :3].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Thresholding
        _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area>1000:
                x,y,w,h=cv2.boundingRect(c)
                roi=img[y:y+h,x:x+w]
                roi=cv2.resize(roi,(20,20))
                roi=cv2.copyMakeBorder(roi,4,4,4,4,cv2.BORDER_CONSTANT,value=0)
                roi=roi.reshape(1,28,28,1)
                roi=roi/255.0
                digits.append(roi)
        for digit in digits:
            prediction=model.predict(digit)
            st.write(str(np.argmax(prediction)))        
