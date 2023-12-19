# train.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, feature_extractor, tokenizer):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Load and process image
        response = requests.get(self.dataframe.iloc[idx]['url'])
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_tensor = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()

        # Tokenize caption
        caption = self.dataframe.iloc[idx]['caption']
        caption_tensor = self.tokenizer(caption, return_tensors="pt", padding='max_length', truncation=True, max_length=128).input_ids.squeeze()

        return image_tensor, caption_tensor

# Load dataset
df = pd.read_csv('your_dataset.csv')
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
dataset = CustomDataset(df, feature_extractor, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)

# Training
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for images, captions in train_loader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images, labels=captions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(pixel_values=images, labels=captions)
            loss = outputs.loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

