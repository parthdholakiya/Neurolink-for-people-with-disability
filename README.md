
# (part-1) Image_Captioning System

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

# (part-2) Audio_Transcription 

### Overview

This code demonstrates how to perform speech recognition using the Hugging Face Whisper model. It includes loading the dataset, preparing the data, training the model, and evaluating its performance.

### Dataset

The dataset used for training is "speechcolab/gigaspeech". To save time and memory, only 1% of the data is used. The dataset is loaded using the load_dataset function from the Hugging Face datasets library.

### Model

The model used is the Whisper model for speech recognition. The model is loaded using the WhisperForConditionalGeneration class from Hugging Face's Transformers library. Additionally, a feature extractor and tokenizer are initialized using WhisperFeatureExtractor and WhisperTokenizer.

### Data Preparation

The audio data is loaded and resampled from 48kHz to 16kHz, which is the sampling rate expected by the Whisper model. The input features are computed using the feature extractor, and the transcriptions are encoded to label ids using the tokenizer.

### Training

The training configuration includes setting the output directory, batch size, learning rate, number of training epochs, and other parameters. The model is trained using the Seq2SeqTrainer class from Hugging Face's Transformers library.

### Evaluation

The evaluation metric used is the word error rate (WER), which is computed using the evaluate library. The compute_metrics function calculates the WER between the predicted and reference transcriptions.

### Post-processing

Some post-processing is applied to the model to enable training, including freezing all layers except the last 5 layers and casting certain layers to float32 for stability. Additionally, Low-Rank Adapter (LoRA) is applied to the model for efficient training.
