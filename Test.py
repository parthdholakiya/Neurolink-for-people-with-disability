# predict.py
import torch
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.load_state_dict(torch.load('model.pth'))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Image URL
image_url = 'your_image_url_here'

# Load and process the image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image_tensor = feature_extractor(images=image, return_tensors="pt").pixel_values

# Generate caption
with torch.no_grad():
    output_ids = model.generate(image_tensor)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Caption:", caption)
