import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from collections import Counter
import pickle
from image_captioning import ImageCaptioningModel, TransformerCaptionModel

class CaptionDataset(Dataset):
    """Dataset class for image captioning"""
    
    def __init__(self, image_dir, captions_file, vocab, transform=None, max_length=20):
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.max_length = max_length
        
        # Load captions
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Create image-caption pairs
        self.samples = []
        for item in self.captions_data:
            image_path = os.path.join(image_dir, item['image'])
            for caption in item['captions']:
                self.samples.append((image_path, caption))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Convert caption to token IDs
        tokens = self.caption_to_tokens(caption)
        
        return image, torch.tensor(tokens, dtype=torch.long)
    
    def caption_to_tokens(self, caption):
        """Convert caption text to token IDs"""
        tokens = ['<start>']
        tokens.extend(caption.lower().split())
        tokens.append('<end>')
        
        # Convert to IDs and pad/truncate
        token_ids = []
        for token in tokens[:self.max_length]:
            token_ids.append(self.vocab.get(token, self.vocab['<unk>']))
        
        # Pad if necessary
        while len(token_ids) < self.max_length:
            token_ids.append(self.vocab['<pad>'])
        
        return token_ids

def build_vocabulary(captions_file, min_freq=2):
    """Build vocabulary from captions"""
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    # Count word frequencies
    word_freq = Counter()
    for item in captions_data:
        for caption in item['captions']:
            words = caption.lower().split()
            word_freq.update(words)
    
    # Create vocabulary
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images, captions = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Pad captions to same length
    captions = torch.stack(captions, 0)
    
    return images, captions

class CaptioningTrainer:
    """Training class for image captioning models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            if isinstance(self.model, TransformerCaptionModel):
                # For transformer model, we need to handle differently
                outputs = self.model(images, captions)
                loss = outputs.loss if hasattr(outputs, 'loss') else criterion(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    captions.view(-1)
                )
            else:
                # For LSTM model
                outputs = self.model(images, captions[:, :-1])
                targets = captions[:, 1:]
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(self.device)
                captions = captions.to(self.device)
                
                if isinstance(self.model, TransformerCaptionModel):
                    outputs = self.model(images, captions)
                    loss = outputs.loss if hasattr(outputs, 'loss') else criterion(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        captions.view(-1)
                    )
                else:
                    outputs = self.model(images, captions[:, :-1])
                    targets = captions[:, 1:]
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=10, lr=1e-3):
        """Complete training loop"""
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print('-' * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_captioning_model.pth')
                print('Saved best model!')

def create_sample_dataset():
    """Create a sample dataset structure for demonstration"""
    sample_data = [
        {
            "image": "sample1.jpg",
            "captions": [
                "A person walking in the park",
                "Someone strolling through green trees",
                "A man walking on a path"
            ]
        },
        {
            "image": "sample2.jpg", 
            "captions": [
                "A red car on the street",
                "Vehicle parked near building",
                "Automobile in urban setting"
            ]
        }
    ]
    
    with open('sample_captions.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created sample_captions.json")
    print("Note: You'll need actual images in an 'images/' directory")

def main():
    """Main training script"""
    print("Image Captioning Training Script")
    print("=" * 40)
    
    # Create sample dataset if it doesn't exist
    if not os.path.exists('sample_captions.json'):
        create_sample_dataset()
    
    # Build vocabulary
    if not os.path.exists('vocab.pkl'):
        print("Building vocabulary...")
        vocab = build_vocabulary('sample_captions.json')
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary built with {len(vocab)} words")
    else:
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (you'll need actual image directory)
    if os.path.exists('images/'):
        train_dataset = CaptionDataset('images/', 'sample_captions.json', 
                                     vocab, transform)
        
        # Split into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, 
                              shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        model = ImageCaptioningModel(
            embed_size=256,
            hidden_size=512,
            vocab_size=len(vocab),
            num_layers=2
        )
        
        # Train model
        trainer = CaptioningTrainer(model)
        trainer.train(train_loader, val_loader, num_epochs=10)
        
    else:
        print("Images directory not found!")
        print("Please create an 'images/' directory with your training images")
        print("and ensure the image filenames match those in sample_captions.json")

if __name__ == "__main__":
    main()