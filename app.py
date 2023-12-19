# app.py
import streamlit as st
import torch
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
import time

# Function to load the model (to avoid reloading on each user interaction)
@st.cache_resource 
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Load the model and tokenizer
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Streamlit app layout
st.title("Image Captioning with GPT-2")
st.write("Upload an image to generate a descriptive caption.")

# Image upload section
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Caption generation button
    if st.button('Generate Caption'):
        with st.spinner('Generating caption...'):
            # Process the image and generate caption
            image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values
            with torch.no_grad():
                output_ids = model.generate(image_tensor)
                caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            time.sleep(2)  # Simulate processing time
        st.success('Done!')
        st.markdown(f"**Generated Caption:** {caption}")
