import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

st.title("Generate Synthetic Image using GAN - GEN AI")

with st.sidebar:
    selected = option_menu("Menu", ["GTA", "Gen Colour"], 
        icons=['house', 'gear'], default_index=0)

if selected == "GTA":
    st.write("Here using Gen AI, we can generate GTA image from a normal image.")
    uploaded_file = st.file_uploader("Upload your color image file here...", type=['png', 'jpeg', 'jpg'])
    model = tf.keras.models.load_model('nor_to_commmmi_pixtopix_gen_6epoch.h5')
    
    def preprocess_input_image(input_image):
        input_image = load_img(input_image, target_size=(256, 256))
        input_array = img_to_array(input_image)
        input_array = (input_array - 127.5) / 127.5  # Normalize to the range [-1, 1]
        input_array = np.expand_dims(input_array, axis=0)
        return input_array
    
    def generate_output_image(input_array):
        translated_image = model.predict(input_array)
        translated_image = 0.5 * translated_image + 0.5  # Denormalize the output image
        return translated_image
    
    if uploaded_file is not None:
        input_array = preprocess_input_image(uploaded_file)
        translated_image = generate_output_image(input_array)
        
        st.write("Real normal image")
        st.image(uploaded_file, width=400)
        st.write("Generated GTA-style image")
        st.image(translated_image[0], width=400)  # Displaying the generated image
else:
    st.write("Here using Gen AI, we can generate a normal image from a thermal image.")
