# Install dependencies (Run manually in Replit shell, not here)
# pip install transformers Pillow torch streamlit

import os
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import torch

# === Hugging Face Token (set as env variable or use st.secrets in production) ===

os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

st.title("🍳 AI Recipe Analyzer")

# === Image Upload ===
uploaded_file = st.file_uploader("Upload a food image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # === BLIP Caption Generation ===
    with st.spinner("Generating caption..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
    st.success("Caption generated!")
    st.write(f"**📝 Caption:** {caption}")

    # === QA Pipeline Setup ===
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    # === Buttons for Questions ===
    col1, col2 = st.columns(2)
    with col1:
        if st.button("What are the ingredients?"):
            with st.spinner("Extracting ingredients..."):
                result = qa_pipeline(question="What are the ingredients?", context=caption)
                st.write(f"**Ingredients:** {result['answer']}")
    with col2:
        if st.button("What are the cooking actions?"):
            with st.spinner("Extracting cooking actions..."):
                result = qa_pipeline(question="What are the cooking actions?", context=caption)
                st.write(f"**Cooking Actions:** {result['answer']}")
else:
    st.info("Please upload an image to get started.")
