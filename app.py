import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Title
st.title("Fashion MNIST Classification with Convolutional Neural Networks")

# Description
st.markdown("""
             This Streamlit app demonstrates a simple Convolutional Neural Network (CNN) model for classifying 
             fashion images from the Fashion MNIST dataset. The model is trained on a subset of the Fashion MNIST 
             dataset and then evaluated on a separate test set. Additionally, the trained model is saved for future use.
             """)

# Load the Fashion Classification CNN model with error handling
@st.cache(allow_output_mutation=True)
def load_fashion_model():
    try:
        model = load_model('fashion_classification_cnn_model.h5')
        return model
    except OSError:
        st.error(
            "Error: Unable to load the model. Please check if the model file exists.")

# Define class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to generate a random fashion image
def generate_random_image():
    # Generate a random image array
    random_image = np.random.rand(28, 28)  # Assuming 28x28 grayscale images

    # Reshape the image for compatibility with the model
    img = random_image.reshape(1, 28, 28, 1)

    return img

# Function to classify a random fashion image
def classify_image(model):
    # Generate a random image
    img = generate_random_image()

    try:
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Get the class label
        class_label = class_labels[predicted_class]

        return img, class_label

    except Exception as e:
        logging.error("Error occurred during image classification: %s", str(e))
        return None, None

# Streamlit app
def main():
    st.title('Fashion Classification')
    st.write('This app allows you to classify fashion items without uploading images.')

    # Load model
    model = load_fashion_model()

    # Add a button to classify a random fashion image
    if st.button('Classify Random Fashion Image'):
        with st.spinner('Classifying...'):
            img, class_label = classify_image(model)
            if img is not None and class_label is not None:
                # Display the random image
                st.image(img.reshape(
                    (28, 28)), caption=f'Predicted Class: {class_label}', use_column_width=True)
            else:
                st.error("An error occurred during image classification. Please try again.")

    # Display Source Code
    if st.button("Show Source Code"):
        st.text("Source Code:")
        st.code("""
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Title
st.title("Fashion MNIST Classification with Convolutional Neural Networks")

# Description
st.markdown("""
             This Streamlit app demonstrates a simple Convolutional Neural Network (CNN) model for classifying 
             fashion images from the Fashion MNIST dataset. The model is trained on a subset of the Fashion MNIST 
             dataset and then evaluated on a separate test set. Additionally, the trained model is saved for future use.
             """)

# Load the Fashion Classification CNN model with error handling
@st.cache(allow_output_mutation=True)
def load_fashion_model():
    try:
        model = load_model('fashion_classification_cnn_model.h5')
        return model
    except OSError:
        st.error(
            "Error: Unable to load the model. Please check if the model file exists.")

# Define class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to generate a random fashion image
def generate_random_image():
    # Generate a random image array
    random_image = np.random.rand(28, 28)  # Assuming 28x28 grayscale images

    # Reshape the image for compatibility with the model
    img = random_image.reshape(1, 28, 28, 1)

    return img

# Function to classify a random fashion image
def classify_image(model):
    # Generate a random image
    img = generate_random_image()

    try:
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Get the class label
        class_label = class_labels[predicted_class]

        return img, class_label

    except Exception as e:
        logging.error("Error occurred during image classification: %s", str(e))
        return None, None

# Streamlit app
def main():
    st.title('Fashion Classification')
    st.write('This app allows you to classify fashion items without uploading images.')

    # Load model
    model = load_fashion_model()

    # Add a button to classify a random fashion image
    if st.button('Classify Random Fashion Image'):
        with st.spinner('Classifying...'):
            img, class_label = classify_image(model)
            if img is not None and class_label is not None:
                # Display the random image
                st.image(img.reshape(
                    (28, 28)), caption=f'Predicted Class: {class_label}', use_column_width=True)
            else:
                st.error("An error occurred during image classification. Please try again.")

    # Display Source Code
    if st.button("Show Source Code"):
        st.text("Source Code:")
        st.code("""
# Paste your source code here
        """)

# Run the Streamlit app
if __name__ == '__main__':
    main()
        """)
        
# Run the Streamlit app
if __name__ == '__main__':
    main()
