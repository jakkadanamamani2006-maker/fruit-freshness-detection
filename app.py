import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(
    page_title="FreshSense AI",
    page_icon="🍎",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #f4fff4,
        #e8f5e9,
        #ffffff
    );
}

/* Floating fruits */
.fruit{
    position:fixed;
    font-size:35px;
    animation: float 10s linear infinite;
    opacity:0.3;
}

@keyframes float{
    0%{
        transform:translateY(100vh);
    }
    100%{
        transform:translateY(-100vh);
    }
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:#2E7D32;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:#555;
}

</style>
""", unsafe_allow_html=True)

# Floating fruits

st.markdown("""
<div class="fruit" style="left:5%">🍎</div>
<div class="fruit" style="left:20%">🍌</div>
<div class="fruit" style="left:40%">🍊</div>
<div class="fruit" style="left:60%">🍎</div>
<div class="fruit" style="left:80%">🍌</div>
""", unsafe_allow_html=True)

# Header

st.markdown("""
<div class="title">
🍎 FreshSense AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI Powered Fruit Freshness Detection
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- MODEL DOWNLOAD ----------------

MODEL_PATH = "fruit_freshness_model.keras"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... Please wait.")

    gdown.download(
        "https://drive.google.com/uc?id=1JHax46671U2VmEyEfqIP51B5bySHLEzb",
        MODEL_PATH,
        quiet=False
    )

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "apple",
    "banana",
    "orange",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

uploaded_file = st.file_uploader(
    "📤 Upload Fruit Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack(
            [img_array] * 3,
            axis=-1
        )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction) * 100

    if "rotten" in predicted_class:

        fruit = predicted_class.replace(
            "rotten",
            ""
        )

        condition = "Rotten ❌"

    else:

        fruit = predicted_class

        condition = "Fresh ✅"

    st.markdown("## Prediction Result")

    st.success(
        f"🍓 Fruit : {fruit.capitalize()}"
    )

    st.success(
        f"🌿 Condition : {condition}"
    )

    st.success(
        f"🎯 Confidence : {confidence:.2f}%"
    )

    st.progress(
        int(confidence)
    )

st.write("---")

st.markdown("""
<center>
<h4>
Developed using CNN + TensorFlow + Streamlit
</h4>
</center>
""", unsafe_allow_html=True)