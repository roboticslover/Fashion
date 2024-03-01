import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the Fashion Classification CNN model with error handling
@st.cache(allow_output_mutation=True)
def load_fashion_model():
    try:
        model = load_model('fashion_classification_cnn_model.h5')
        return model
    except OSError:
        st.error("Error: Unable to load the model. Please check if the model file exists.")

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

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    # Get the class label
    class_label = class_labels[predicted_class]

    return img, class_label

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
            # Display the random image
            st.image(img.reshape((28, 28)), caption=f'Predicted Class: {class_label}', use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
