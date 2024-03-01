import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

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
    return np.random.rand(1, 28, 28, 1)  # Assuming 28x28 grayscale images

# Function to classify a random fashion image
def classify_image(model):
    img = generate_random_image()
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
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
            # Resize the image for better display
            resized_img = Image.fromarray((img.reshape((28, 28)) * 255).astype(np.uint8)).resize((150, 150))
            # Display the resized image
            st.image(resized_img, caption=f'Predicted Class: {class_label}', use_column_width=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
