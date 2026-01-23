import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import requests
from io import BytesIO

class ImageEncoder(nn.Module):
    """Extract features from images using pre-trained CNN"""
    
    def __init__(self, embed_size=256, model_type='resnet50'):
        super(ImageEncoder, self).__init__()
        
        if model_type == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            modules = list(resnet.children())[:-1]  # Remove final FC layer
            self.resnet = nn.Sequential(*modules)
            self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        elif model_type == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            self.vgg = vgg.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.embed = nn.Linear(512 * 7 * 7, embed_size)
        
        self.model_type = model_type
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        if self.model_type == 'resnet50':
            with torch.no_grad():
                features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
        elif self.model_type == 'vgg16':
            with torch.no_grad():
                features = self.vgg(images)
                features = self.avgpool(features)
            features = features.reshape(features.size(0), -1)
            
        features = self.dropout(features)
        features = self.embed(features)
        return features

class CaptionDecoder(nn.Module):
    """Generate captions using LSTM/GRU"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(CaptionDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, caption_length)
        
        batch_size = features.size(0)
        
        # Project features to hidden size if needed
        if features.size(1) != self.hidden_size:
            features_projected = nn.Linear(features.size(1), self.hidden_size).to(features.device)(features)
        else:
            features_projected = features
        
        # Initialize hidden state with projected image features
        h0 = features_projected.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Embed captions
        embeddings = self.embedding(captions)
        
        # Concatenate image features as first input
        features_input = features.unsqueeze(1)  # (batch_size, 1, embed_size)
        embeddings = torch.cat((features_input, embeddings), dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(self.dropout(lstm_out))
        
        return outputs
    
    def generate_caption(self, features, max_length=20, vocab=None):
        """Generate caption for given image features"""
        batch_size = features.size(0)
        
        # Project features to hidden size if needed
        if features.size(1) != self.hidden_size:
            features_projected = nn.Linear(features.size(1), self.hidden_size).to(features.device)(features)
        else:
            features_projected = features
        
        # Initialize hidden state
        h = features_projected.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)
        
        # Start with image features
        inputs = features.unsqueeze(1)
        captions = []
        
        for i in range(max_length):
            lstm_out, (h, c) = self.lstm(inputs, (h, c))
            outputs = self.linear(lstm_out)
            predicted = outputs.argmax(dim=2)
            
            captions.append(predicted.item())
            
            # Use predicted word as next input
            inputs = self.embedding(predicted)
            
            # Stop if end token is generated
            if vocab and predicted.item() == vocab.get('<end>', 1):
                break
                
        return captions

class ImageCaptioningModel(nn.Module):
    """Complete Image Captioning Model"""
    
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=10000, 
                 num_layers=2, cnn_model='resnet50'):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = ImageEncoder(embed_size, cnn_model)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size, num_layers)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, images, max_length=20, vocab=None):
        features = self.encoder(images)
        return self.decoder.generate_caption(features, max_length, vocab)

class TransformerCaptionModel(nn.Module):
    """Transformer-based caption generation using GPT-2"""
    
    def __init__(self, embed_size=768):
        super(TransformerCaptionModel, self).__init__()
        
        self.encoder = ImageEncoder(embed_size, 'resnet50')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load pre-trained GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Project image features to GPT-2 embedding space
        self.feature_projection = nn.Linear(embed_size, self.gpt2.config.n_embd)
        
    def forward(self, images, captions=None):
        # Extract image features
        image_features = self.encoder(images)
        image_embeddings = self.feature_projection(image_features).unsqueeze(1)
        
        if captions is not None:
            # Training mode
            inputs = self.tokenizer(captions, return_tensors='pt', 
                                  padding=True, truncation=True)
            
            # Get text embeddings
            text_embeddings = self.gpt2.transformer.wte(inputs['input_ids'])
            
            # Concatenate image and text embeddings
            combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
            
            # Forward through GPT-2
            outputs = self.gpt2(inputs_embeds=combined_embeddings, 
                              attention_mask=inputs['attention_mask'])
            return outputs
        else:
            # Inference mode
            return self.generate_caption(image_embeddings)
    
    def generate_caption(self, image_embeddings, max_length=50):
        """Generate caption using beam search or greedy decoding"""
        batch_size = image_embeddings.size(0)
        
        # Start with image embeddings
        generated_ids = torch.tensor([[self.tokenizer.bos_token_id]] * batch_size)
        
        for _ in range(max_length):
            # Get text embeddings for generated tokens
            text_embeddings = self.gpt2.transformer.wte(generated_ids)
            
            # Combine with image embeddings
            combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.gpt2(inputs_embeds=combined_embeddings)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            
            # Stop if EOS token is generated
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode to text
        captions = []
        for ids in generated_ids:
            caption = self.tokenizer.decode(ids[1:], skip_special_tokens=True)  # Skip BOS
            captions.append(caption)
        
        return captions

class ImageCaptioningPipeline:
    """Complete pipeline for image captioning"""
    
    def __init__(self, model_type='lstm', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_type = model_type
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        if model_type == 'lstm':
            self.model = ImageCaptioningModel()
        elif model_type == 'transformer':
            self.model = TransformerCaptionModel()
        
        self.model.to(device)
        
    def load_image(self, image_path_or_url):
        """Load image from file path or URL"""
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        
        return image
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = self.load_image(image)
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def generate_caption(self, image_path_or_url, max_length=20):
        """Generate caption for an image"""
        self.model.eval()
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path_or_url)
        
        with torch.no_grad():
            if self.model_type == 'lstm':
                caption_ids = self.model.generate_caption(image_tensor, max_length)
                # Convert IDs to words (requires vocabulary)
                caption = "Generated caption (requires trained vocabulary)"
            elif self.model_type == 'transformer':
                captions = self.model.generate_caption(
                    self.model.feature_projection(
                        self.model.encoder(image_tensor)
                    ).unsqueeze(1), 
                    max_length
                )
                caption = captions[0]
        
        return caption
    
    def visualize_result(self, image_path_or_url, caption):
        """Display image with generated caption"""
        image = self.load_image(image_path_or_url)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Caption: {caption}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

# Example usage and training utilities
def create_sample_vocabulary():
    """Create a sample vocabulary for demonstration"""
    vocab = {
        '<start>': 0, '<end>': 1, '<unk>': 2, '<pad>': 3,
        'a': 4, 'an': 5, 'the': 6, 'is': 7, 'are': 8, 'in': 9, 'on': 10,
        'person': 11, 'man': 12, 'woman': 13, 'child': 14, 'dog': 15, 'cat': 16,
        'car': 17, 'bike': 18, 'tree': 19, 'house': 20, 'building': 21,
        'walking': 22, 'running': 23, 'sitting': 24, 'standing': 25,
        'red': 26, 'blue': 27, 'green': 28, 'black': 29, 'white': 30
    }
    return vocab

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """Training loop for the captioning model"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(model.device)
            captions = captions.to(model.device)
            
            # Forward pass
            outputs = model(images, captions[:, :-1])  # Exclude last token
            targets = captions[:, 1:]  # Exclude first token
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), 
                           targets.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    # Example usage
    print("Image Captioning AI Demo")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = ImageCaptioningPipeline(model_type='transformer')
    
    # Example with a sample image URL
    sample_image_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500"
    
    try:
        caption = pipeline.generate_caption(sample_image_url)
        print(f"Generated Caption: {caption}")
        
        # Visualize result
        pipeline.visualize_result(sample_image_url, caption)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This is a demonstration. For full functionality, ")
        print("you would need to train the model on a captioning dataset.")