
# (part-1) GPT-2 Based Image-to-Text System

![ImageTotext-video](https://github.com/parthdholakiya/GPT-2-Based-Image-to-Text-System/assets/94167271/13fb6eb4-4e75-4502-98f0-a3991dd337cd)

## Image To Text Demo on GPT-2

This repository contains a demo showcasing how to use the GPT-2 model to generate text captions from images. The demo utilizes the Hugging Face `transformers` library for model loading and processing.

### Requirements

To run the demo, ensure you have the following dependencies installed:
#### fastparquet pandas matplotlib scikit-learn transformers torch


### Data Set

The demo uses a subset of the 220k GPT-4 Vision Captions from the LVIS dataset. The dataset contains image URLs along with their corresponding captions. Due to memory constraints, only a small portion of the data is used in this demo.

### Model Loading

The demo loads the GPT-2 model using the ViTFeatureExtractor, AutoTokenizer, and VisionEncoderDecoderModel classes from the transformers library. The model is then moved to the GPU if available.

### Training Loop

The training loop fine-tunes the GPT-2 model on the image-caption dataset. It unfreezes the last 5 layers of the model for training while keeping the rest of the layers frozen. The loop runs for 10 epochs, optimizing the model with the Adam optimizer.

### Generate Caption Function

A function is provided to generate captions for new images using the trained model. The function loads a pre-trained model state and sets the model to evaluation mode before generating captions.

### (part-2)
