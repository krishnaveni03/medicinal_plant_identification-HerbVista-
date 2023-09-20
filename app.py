import streamlit as st
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub

# Load the TensorFlow Hub model
IMAGE_SHAPE = (224, 224)
model = tf.keras.Sequential([hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1', input_shape=IMAGE_SHAPE + (3,))])

# Load labels
with open('labelplants.txt', 'r') as f:
    labels = f.read().splitlines()

st.title('Image Classification with TensorFlow Hub')

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    result = model.predict(image)

    if len(labels) > 0:
        predicted_index = np.argmax(result)
        if predicted_index < len(labels) and predicted_index >= 0:
            predicted_label = labels[predicted_index]
            st.write(f"Prediction: {predicted_label}")
            
        else:
            st.write("Invalid prediction index")
    else:
        st.write("No labels found. Please check the 'labelplants.txt' file.")
