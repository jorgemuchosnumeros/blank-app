import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.layers import TextVectorization
from PIL import Image

st.set_page_config(page_title="Model Selector", layout="centered")
st.title("Model Selector")

# Sidebar selection
model_type = st.sidebar.selectbox(
    "Choose a model:",
    ("Image Classification", "Text Classification", "Regression", "About")
)

if model_type == "Image Classification":
    st.subheader("🖼️ Image Classification (Sign Language MNIST)")

    # Load model
    try:
        model = tf.keras.models.load_model("model_image.keras")
        st.success("Model loaded successfully.")
    except:
        st.error("❌ Could not find: model_image.keras")
        st.stop()


    # ---------- IMAGE CLASSIFICATION ----------
    uploaded_file = st.file_uploader("Upload a PNG image (28x28 or any size)", type=["png"])

    if uploaded_file is not None:
        # Open, convert to grayscale, resize
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Uploaded image", width=150)

        # Convert to NumPy and normalize
        img_array = np.array(image) / 255.0  # scale to [0, 1]
        flat = img_array.flatten().reshape(1, -1)

        # Optional: preview as DataFrame
        df = pd.DataFrame(flat, columns=[f"px{i}" for i in range(784)])
        st.write("Flattened grayscale input (shape: 1x784):")
        st.dataframe(df)

        # Predict
        pred = model.predict(flat)[0]
        label_idx = np.argmax(pred)
        confidence = np.max(pred)

        # Map back to A-Z (excluding J and Z in sign language dataset)
        import string
        label_map = list(string.ascii_uppercase)
        predicted_letter = label_map[label_idx]

        st.markdown(f"### 🧠 Predicted Letter: **{predicted_letter}** (Confidence: {confidence:.2f})")


# ---------- TEXT CLASSIFICATION ----------
elif model_type == "Text Classification":
    st.subheader("📝 Text Classification Model")
    try:
        model = tf.keras.models.load_model("model_text.keras")
        st.success("Model loaded successfully.")
    except:
        st.error("❌ Could not find: model_text.keras")
        st.stop()

    # Text input
    user_input = st.text_area("Enter a review or sentence:")

    if st.button("Classify Text"):
        if user_input.strip() == "":
            st.warning("Please enter text.")
        else:
            # Vectorize manually if TextVectorization not embedded
            vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=250)
            df = pd.read_csv("your_training_data.csv")  # Replace with actual CSV if needed
            vectorizer.adapt(df["X"].values)
            vec_input = vectorizer(tf.constant([user_input]))
            pred = model.predict(vec_input)[0][0]
            label = "Positive" if pred > 0.5 else "Negative"
            st.markdown(f"### Prediction: **{label}** ({pred:.2f})")


# ---------- REGRESSION ----------
elif model_type == "Regression":
    st.subheader("📈 House Price Regression Model")

    try:
        model = tf.keras.models.load_model("model_regression.keras")
        scaler = joblib.load("scaler.save")
        st.success("Model and scaler loaded successfully.")
    except:
        st.error("❌ Could not find: model_regression.keras or scaler.save")
        st.stop()

    # User input fields
    st.markdown("Enter house features below:")
    MedInc = st.number_input("Median Income (×10k USD)", value=5.0)
    HouseAge = st.number_input("House Age (years)", value=30.0)
    AveRooms = st.number_input("Average Rooms", value=5.0)
    AveBedrms = st.number_input("Average Bedrooms", value=1.0)
    Population = st.number_input("Population", value=300.0)
    AveOccup = st.number_input("Average Occupants", value=2.0)
    Latitude = st.number_input("Latitude", value=37.0)
    Longitude = st.number_input("Longitude", value=-122.0)

    if st.button("Predict Price"):
        
        input_data = pd.DataFrame([{
            "MedInc": MedInc,
            "HouseAge": HouseAge,
            "AveRooms": AveRooms,
            "AveBedrms": AveBedrms,
            "Population": Population,
            "AveOccup": AveOccup,
            "Latitude": Latitude,
            "Longitude": Longitude
        }])
        scaled = scaler.transform(input_data)
        scaled = np.array(scaled)
        print("Input shape:", scaled.shape)
        pred = model.predict(scaled)[0][0]
        st.markdown(f"### 🏠 Predicted House Price: **${pred * 100_000:,.2f}**")


elif model_type == "About":
    st.subheader("📚 About This App")
    st.markdown("""
    Welcome to my Class Activity #2!
    This interface allows you to interact with three different machine learning models:

    ### 🧠 Models Available:
    - **Image Classification**  
      Upload a 28×28 grayscale image (like a handwritten sign) and get a predicted letter (A–Z).
    
    - **Text Classification**  
      Enter a sentence or review and get a binary sentiment prediction (e.g., Positive/Negative).
    
    - **Regression**  
      Input housing-related data and get a predicted house price based on California housing patterns.


    ---
    _Built by Jorge — 2025_
    """)

