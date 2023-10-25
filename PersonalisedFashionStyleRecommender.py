import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


imagenames = pickle.load(open('imagenames.pkl', 'rb'))
features_list = np.array(pickle.load(open('features_list.pkl', 'rb')))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.title('Personalised Fashion Styles Recommender')
st.subheader('By - Umang Tilokani')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, features_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# File Upload and Save
uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Displaying the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # Feature extraction from image/ Feature Engineering
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # Recommending Fashion Styles
        st.subheader('Recommended Styles')
        indices = recommend(features, features_list)
        # Displaying Fashion Styles
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(imagenames[indices[0][0]], width=100)
        with col2:
            st.image(imagenames[indices[0][1]], width=100)
        with col3:
            st.image(imagenames[indices[0][2]], width=100)
        with col4:
            st.image(imagenames[indices[0][3]], width=100)
        with col5:
            st.image(imagenames[indices[0][4]], width=100)
    else:
        st.header("Some error occurred in file upload.")
