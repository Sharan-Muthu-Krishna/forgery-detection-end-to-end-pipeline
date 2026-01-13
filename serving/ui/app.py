import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Forgery Detection", layout="centered")

st.title("🕵️ Forgery Detection System")
st.markdown("Upload an image to check whether it is **Original** or **Forged**.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            files = {
                "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
            }

            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()

                label = result["prediction"]
                confidence = result["confidence"] 

                if label == "Original":
                    st.success(f"✅ Original — Confidence: {confidence}%")
                else:
                    st.error(f"❌ Forged — Confidence: {confidence}%")
            else:
                st.error("❌ API Error")
                st.text(response.text)
