import streamlit as st
import tensorflow as tf
import base64
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Define a function for model inference
def predict(image):
    # Open and preprocess the image
    img = Image.open(image).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((256, 256))  # Resize the image to match the model input size
    img_array = np.array(img)  # Convert the image to a NumPy array
    img_array = img_array / 255.0  # Normalize pixel values to the range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make predictions using the loaded model
    predictions = model.predict(img_array)

    return predictions

# Streamlit app code
st.set_page_config(
    page_title="Malaria Detection App",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("Malaria Detection")
st.sidebar.write(
    "This app uses a deep learning model to predict whether an uploaded blood smear image is normal or infected with malaria."
)

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
# Main content
st.title("Malaria Detection App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image with border
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True, output_format="JPEG", width=300, border_radius=10)

    # Perform prediction
    if st.button("Predict"):
        result = predict(uploaded_file)

        # Display the prediction results
        st.write("Prediction Results:")
        prediction_label = "Normal" if result >= 0.5 else "Infected"
        st.success(f"The image is predicted as {prediction_label}")

# Footer
st.markdown("---")
st.write("Developed by Atharva Chavan")
st.write("Copyright © 2023. All rights reserved.")
