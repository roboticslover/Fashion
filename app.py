import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image

# Load the Fashion Classification CNN model efficiently
@st.cache(allow_output_mutation=True)  # Cache the model for faster loading
def load_model():
    try:
        model = keras_load_model('fashion_classification_cnn_model.h5')
        return model
    except FileNotFoundError:
        st.error("Error: Unable to load the model. Please check if the model file exists.")
        return None  # Return None to indicate model loading failure

# Define class labels clearly
class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Generate a random fashion image concisely
def generate_random_image():
    return np.random.rand(1, 28, 28, 1)

# Classify a random fashion image efficiently
def classify_image(model):
    img = generate_random_image()
    prediction = model.predict(img)[0]  # Access the prediction for the first image
    predicted_class = np.argmax(prediction)
    class_label = class_labels[predicted_class]
    return img, class_label

# Create a visually appealing and informative Streamlit app
def main():
    st.title('Fashion Classification ')
    st.write('This app allows you to classify fashion items effortlessly!')

    # Load the model (or handle potential errors)
    model = load_model()
    if model is None:
        return  # Exit if model loading failed

    # Add a visually engaging button for image classification
    if st.button('Let\'s Classify a Fashion Item!'):
        with st.spinner('Working on your request... ⏱️'):
            img, class_label = classify_image(model)
            resized_img = Image.fromarray((img.reshape((28, 28)) * 255).astype(np.uint8)).resize((150, 150))
            st.image(resized_img, caption=f'Predicted Class: {class_label}', use_column_width=True)

if __name__ == '__main__':
    main()
