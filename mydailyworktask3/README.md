# Image Captioning AI

A comprehensive image captioning system that combines computer vision and natural language processing to automatically generate descriptive captions for images.

## Features

- **Multiple Model Architectures**: Support for both LSTM-based and Transformer-based (GPT-2) caption generation
- **Pre-trained CNN Encoders**: Uses ResNet50 and VGG16 for robust image feature extraction
- **Flexible Pipeline**: Easy-to-use interface for both training and inference
- **Comprehensive Training**: Complete training pipeline with validation and model checkpointing
- **Interactive Demo**: Multiple demo modes for testing and evaluation

## Architecture

### Image Encoder
- **ResNet50/VGG16**: Pre-trained CNN models for feature extraction
- **Feature Projection**: Linear layer to map CNN features to embedding space
- **Dropout**: Regularization to prevent overfitting

### Caption Decoder Options

#### 1. LSTM-based Decoder
- Multi-layer LSTM for sequential caption generation
- Attention mechanism between image features and text
- Vocabulary-based word generation

#### 2. Transformer-based Decoder
- GPT-2 integration for advanced language modeling
- Image features as context for text generation
- Beam search and greedy decoding options

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from image_captioning import ImageCaptioningPipeline

# Initialize pipeline
pipeline = ImageCaptioningPipeline(model_type='transformer')

# Generate caption for an image
caption = pipeline.generate_caption('path/to/image.jpg')
print(f"Caption: {caption}")

# Visualize result
pipeline.visualize_result('path/to/image.jpg', caption)
```

### Run Demo

```bash
python demo.py
```

Choose from multiple demo options:
1. Basic demo with sample images
2. Model comparison (LSTM vs Transformer)
3. Interactive demo with custom URLs
4. Performance benchmark
5. Run all demos

## Training

### Prepare Dataset

Create a JSON file with image-caption pairs:

```json
[
  {
    "image": "image1.jpg",
    "captions": [
      "A person walking in the park",
      "Someone strolling through trees"
    ]
  }
]
```

### Train Model

```bash
python train_captioning.py
```

The training script will:
- Build vocabulary from captions
- Create train/validation splits
- Train the model with proper validation
- Save the best model checkpoint

### Custom Training

```python
from train_captioning import CaptioningTrainer
from image_captioning import ImageCaptioningModel

# Initialize model
model = ImageCaptioningModel(
    embed_size=256,
    hidden_size=512,
    vocab_size=10000
)

# Train
trainer = CaptioningTrainer(model)
trainer.train(train_loader, val_loader, num_epochs=20)
```

## Model Components

### ImageEncoder
```python
encoder = ImageEncoder(embed_size=256, model_type='resnet50')
features = encoder(images)  # Extract image features
```

### CaptionDecoder (LSTM)
```python
decoder = CaptionDecoder(embed_size=256, hidden_size=512, vocab_size=10000)
captions = decoder.generate_caption(features, max_length=20)
```

### TransformerCaptionModel
```python
model = TransformerCaptionModel(embed_size=768)
captions = model.generate_caption(image_embeddings, max_length=50)
```

## Configuration Options

### Model Parameters
- `embed_size`: Dimension of feature embeddings (default: 256)
- `hidden_size`: LSTM hidden state size (default: 512)
- `vocab_size`: Vocabulary size (default: 10000)
- `num_layers`: Number of LSTM layers (default: 2)

### Training Parameters
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 1e-3)
- `num_epochs`: Training epochs (default: 10)
- `max_length`: Maximum caption length (default: 20)

## Performance Optimization

### GPU Acceleration
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipeline = ImageCaptioningPipeline(device=device)
```

### Batch Processing
```python
# Process multiple images at once
images = [img1, img2, img3]
captions = [pipeline.generate_caption(img) for img in images]
```

### Model Optimization
- Use mixed precision training for faster training
- Implement gradient clipping to prevent exploding gradients
- Use learning rate scheduling for better convergence

## Evaluation Metrics

Common metrics for image captioning:
- **BLEU Score**: Measures n-gram overlap with reference captions
- **METEOR**: Considers synonyms and paraphrases
- **CIDEr**: Consensus-based metric for image description
- **ROUGE-L**: Longest common subsequence based metric

## Advanced Features

### Attention Visualization
```python
# Visualize attention weights (requires modification)
attention_weights = model.get_attention_weights(image, caption)
visualize_attention(image, caption, attention_weights)
```

### Beam Search Decoding
```python
# Generate multiple caption candidates
captions = model.beam_search(image_features, beam_size=5)
```

### Fine-tuning
```python
# Fine-tune on domain-specific data
model.load_pretrained('best_captioning_model.pth')
trainer.fine_tune(domain_data_loader, num_epochs=5)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable mixed precision training

2. **Poor Caption Quality**
   - Increase training data
   - Tune hyperparameters
   - Use pre-trained language models

3. **Slow Inference**
   - Use GPU acceleration
   - Optimize model architecture
   - Implement model quantization

### Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- PIL/Pillow for image processing
- matplotlib for visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Pre-trained models from torchvision and Hugging Face
- Inspiration from Show, Attend and Tell paper
- GPT-2 integration for advanced language modeling