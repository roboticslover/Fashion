import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the Fashion Classification CNN model with error handling
try:
    model = load_model('fashion_classification_cnn_model.h5')
except OSError:
    st.error("Error: Unable to load the model. Please check if the model file exists.")
    st.stop()

# Define class labels
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Function to classify a random fashion image


def classify_image():
    # Generate a random index
    random_index = np.random.randint(
        0, len(X_test))  # Assuming X_test is defined

    # Get the image and preprocess it
    img = X_test[random_index]
    img = img.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Get the class label
    class_label = class_labels[predicted_class]

    st.subheader('Fashion Classification Result')
    st.write(
        f'The model has classified the fashion item as: **{class_label}**.')

# Streamlit app


def main():
    st.title('Fashion Classification')
    st.write('This app allows you to classify fashion items without uploading images.')

    # Add a button to classify a random fashion image
    if st.button('Classify Random Fashion Image'):
        classify_image()


# Run the Streamlit app
if __name__ == '__main__':
    main()
