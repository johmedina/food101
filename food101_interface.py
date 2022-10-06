import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import pandas as pd


model = tf.keras.models.load_model('./food101_pruned_model.h5')

# Image Preprocessing
def predict_image(image_data, model):
    size = (256,256)    
    image = ImageOps.fit(image_data, size)
    image = np.asarray(image)
    img_resize = image/255.
    img_reshape = img_resize[np.newaxis,...]

    prediction = model.predict(img_reshape)
    return prediction
        

# Front-end design using the Python library Streamlit
st.set_page_config(layout="wide")

st.title("FOOD 101: Food Classifier Using Deep Learning")

st.write("A web interface for classifying food images from the Food101 dataset using Deep Learning for CPEG660 Fall 2022 Midterm Project")




col1, col2 = st.columns(2, gap="large")

with col1:
    # Upload image prompt
    file = st.file_uploader("Please upload an image file and start predicting!", type=["jpg", "png"])
    if file is None:
        st.text("Please upload an image file")			

    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)

with col2:
    # Output Prediction
    if file is not None:
        prediction = predict_image(image, model)
        
        vert_space = '<div style="padding: 50px 30px;"></div>'
        st.markdown(vert_space, unsafe_allow_html=True)

        # The class with maximum probability is the returned output
        if np.argmax(prediction) == 0:				
            st.write(" ## It is an apple pie!")
        elif np.argmax(prediction) == 1:
            st.write(" ## It is baby back ribs!")
        elif np.argmax(prediction) == 2:
            st.write(" ## It is a baklava!")
        elif np.argmax(prediction) == 3:
            st.write(" ## It is beef carpaccio!")
        elif np.argmax(prediction) == 4:
            st.write(" ## It is beef tartare!")
        elif np.argmax(prediction) == 5:
            st.write(" ## It is a beet salad!")
        elif np.argmax(prediction) == 6:
            st.write(" ## It is beignets!")
        elif np.argmax(prediction) == 7:
            st.write(" ## It is bibimbap!")
        elif np.argmax(prediction) == 8:
            st.write(" ## It is bread pudding!")
        elif np.argmax(prediction) == 9:
            st.write(" ## It is a breakfast burrito!")
        elif np.argmax(prediction) == 10:
            st.write(" ## It is bruschetta!")
        elif np.argmax(prediction) == 11:
            st.write(" ## It is caesar salad!")
        elif np.argmax(prediction) == 12:
            st.write(" ## It is a cannoli!")
        elif np.argmax(prediction) == 13:
            st.write(" ## It is a caprese salad!")
        elif np.argmax(prediction) == 14:
            st.write(" ## It is a carrot cake!")
        elif np.argmax(prediction) == 15:
            st.write(" ## It is ceviche!")
        elif np.argmax(prediction) == 16:
            st.write(" ## It is a cheesecake!")
        elif np.argmax(prediction) == 17:
            st.write(" ## It is a cheese plate!")
        elif np.argmax(prediction) == 18:
            st.write(" ## It is chicken curry!")
        elif np.argmax(prediction) == 19:
            st.write(" ## It is chicken quisadilla!")
        else:
            st.write(" ## This food is classified as others.")

        #Labels:
        df = pd.DataFrame()
        df['probability'] = prediction[0]
        df['label'] = ['apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 'bibimbap', 
            'bread pudding', 'breakfast burritto', 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake', 'ceviche', 'cheesecake',
            'cheese plate', 'chicken curry', 'chicken quisadilla', 'others']
        st.dataframe(df)
