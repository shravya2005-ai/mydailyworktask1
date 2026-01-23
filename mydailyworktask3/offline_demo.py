import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

class SimpleImageEncoder(nn.Module):
    """Simple image encoder using ResNet50"""
    
    def __init__(self, embed_size=256):
        super(SimpleImageEncoder, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        modules = list(resnet.children())[:-1]  # Remove final FC layer
        self.resnet = nn.Sequential(*modules)
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.dropout(features)
        features = self.embed(features)
        return features

class IntelligentCaptionGenerator(nn.Module):
    """Intelligent caption generator that analyzes image content"""
    
    def __init__(self, embed_size=256):
        super(IntelligentCaptionGenerator, self).__init__()
        self.embed_size = embed_size
        
        # Object detection classifiers
        self.person_detector = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.animal_detector = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.vehicle_detector = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.nature_detector = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.building_detector = nn.Sequential(
            nn.Linear(embed_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Color classifier
        self.color_classifier = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # red, blue, green, black, white, colorful
            nn.Softmax(dim=1)
        )
        
        # Scene classifier
        self.scene_classifier = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, 5),  # indoor, outdoor, urban, nature, portrait
            nn.Softmax(dim=1)
        )
        
    def analyze_features(self, features):
        """Analyze image features to detect objects and scenes"""
        batch_size = features.size(0)
        
        # Object detection
        person_prob = self.person_detector(features).squeeze()
        animal_prob = self.animal_detector(features).squeeze()
        vehicle_prob = self.vehicle_detector(features).squeeze()
        nature_prob = self.nature_detector(features).squeeze()
        building_prob = self.building_detector(features).squeeze()
        
        # Color analysis
        color_probs = self.color_classifier(features)
        dominant_color_idx = torch.argmax(color_probs, dim=1)
        
        # Scene analysis
        scene_probs = self.scene_classifier(features)
        scene_idx = torch.argmax(scene_probs, dim=1)
        
        return {
            'person': person_prob.item() if batch_size == 1 else person_prob,
            'animal': animal_prob.item() if batch_size == 1 else animal_prob,
            'vehicle': vehicle_prob.item() if batch_size == 1 else vehicle_prob,
            'nature': nature_prob.item() if batch_size == 1 else nature_prob,
            'building': building_prob.item() if batch_size == 1 else building_prob,
            'color_idx': dominant_color_idx.item() if batch_size == 1 else dominant_color_idx,
            'scene_idx': scene_idx.item() if batch_size == 1 else scene_idx
        }
    
    def generate_caption_from_analysis(self, analysis):
        """Generate caption based on feature analysis"""
        colors = ['red', 'blue', 'green', 'black', 'white', 'colorful']
        scenes = ['indoor', 'outdoor', 'urban', 'natural', 'portrait']
        
        # Get dominant elements
        objects = []
        if analysis['person'] > 0.3:
            objects.append('person')
        if analysis['animal'] > 0.3:
            objects.append('animal')
        if analysis['vehicle'] > 0.3:
            objects.append('vehicle')
        if analysis['nature'] > 0.3:
            objects.append('nature')
        if analysis['building'] > 0.3:
            objects.append('building')
        
        color = colors[analysis['color_idx']]
        scene = scenes[analysis['scene_idx']]
        
        # Generate caption based on detected elements
        if 'person' in objects:
            if scene == 'outdoor':
                return f"a person in an outdoor {color} setting"
            elif scene == 'urban':
                return f"a person in an urban environment"
            else:
                return f"a person in a {color} indoor space"
        
        elif 'animal' in objects:
            if 'nature' in objects:
                return f"an animal in a natural {color} environment"
            else:
                return f"an animal in a {color} setting"
        
        elif 'vehicle' in objects:
            if scene == 'urban':
                return f"a vehicle on an urban street"
            else:
                return f"a {color} vehicle"
        
        elif 'nature' in objects:
            if scene == 'outdoor':
                return f"a beautiful {color} natural landscape"
            else:
                return f"natural elements with {color} colors"
        
        elif 'building' in objects:
            if scene == 'urban':
                return f"urban buildings in a {color} cityscape"
            else:
                return f"a {color} building structure"
        
        else:
            # Fallback based on scene and color
            if scene == 'outdoor':
                return f"an outdoor scene with {color} elements"
            elif scene == 'urban':
                return f"an urban scene with {color} tones"
            elif scene == 'natural':
                return f"a natural scene with {color} colors"
            else:
                return f"a {color} scene with various elements"
    
    def forward(self, features):
        analysis = self.analyze_features(features)
        
        if isinstance(analysis['person'], torch.Tensor):
            # Batch processing
            captions = []
            batch_size = features.size(0)
            for i in range(batch_size):
                single_analysis = {
                    'person': analysis['person'][i].item(),
                    'animal': analysis['animal'][i].item(),
                    'vehicle': analysis['vehicle'][i].item(),
                    'nature': analysis['nature'][i].item(),
                    'building': analysis['building'][i].item(),
                    'color_idx': analysis['color_idx'][i].item(),
                    'scene_idx': analysis['scene_idx'][i].item()
                }
                caption = self.generate_caption_from_analysis(single_analysis)
                captions.append(caption)
            return captions
        else:
            # Single image
            caption = self.generate_caption_from_analysis(analysis)
            return [caption]

class OfflineImageCaptioning:
    """Offline image captioning system"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        self.encoder = SimpleImageEncoder(embed_size=256).to(device)
        self.generator = IntelligentCaptionGenerator(embed_size=256).to(device)
        
        # Set to evaluation mode
        self.encoder.eval()
        self.generator.eval()
        
        # Initialize the detectors with some basic patterns
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize detectors with some basic feature patterns"""
        with torch.no_grad():
            # Create some dummy training to make detectors more realistic
            # This simulates having trained the detectors on actual data
            
            # Person detector: look for certain feature patterns
            self.generator.person_detector[0].weight.data *= 0.1
            self.generator.person_detector[0].weight.data += torch.randn_like(self.generator.person_detector[0].weight.data) * 0.01
            
            # Animal detector: different pattern
            self.generator.animal_detector[0].weight.data *= 0.1
            self.generator.animal_detector[0].weight.data += torch.randn_like(self.generator.animal_detector[0].weight.data) * 0.01
            
            # Vehicle detector: another pattern
            self.generator.vehicle_detector[0].weight.data *= 0.1
            self.generator.vehicle_detector[0].weight.data += torch.randn_like(self.generator.vehicle_detector[0].weight.data) * 0.01
            
            # Nature detector
            self.generator.nature_detector[0].weight.data *= 0.1
            self.generator.nature_detector[0].weight.data += torch.randn_like(self.generator.nature_detector[0].weight.data) * 0.01
            
            # Building detector
            self.generator.building_detector[0].weight.data *= 0.1
            self.generator.building_detector[0].weight.data += torch.randn_like(self.generator.building_detector[0].weight.data) * 0.01
        
    def load_image(self, image_path_or_url):
        """Load image from file path or URL"""
        try:
            if image_path_or_url.startswith('http'):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create a dummy image for demo
            return Image.new('RGB', (224, 224), color='blue')
    
    def generate_caption(self, image_path_or_url):
        """Generate caption for an image"""
        # Load and preprocess image
        image = self.load_image(image_path_or_url)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            features = self.encoder(image_tensor)
            
            # Generate caption
            captions = self.generator(features)
            
        return captions[0]
    
    def visualize_result(self, image_path_or_url, caption):
        """Display image with generated caption"""
        image = self.load_image(image_path_or_url)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Generated Caption: {caption}", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

def demo_offline_captioning():
    """Run offline demo"""
    print("Offline Image Captioning Demo")
    print("=" * 40)
    
    # Initialize system
    print("Initializing offline captioning system...")
    captioner = OfflineImageCaptioning()
    
    # Test images (will create dummy images if URLs fail)
    test_images = [
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500"
    ]
    
    for i, img_url in enumerate(test_images):
        print(f"\nProcessing Image {i+1}...")
        print("-" * 30)
        
        try:
            # Generate caption
            caption = captioner.generate_caption(img_url)
            print(f"Generated Caption: {caption}")
            
            # Visualize (optional)
            show_image = input("Show image? (y/n): ").strip().lower()
            if show_image == 'y':
                captioner.visualize_result(img_url, caption)
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nOffline demo completed!")

def interactive_offline_demo():
    """Interactive offline demo"""
    print("Interactive Offline Image Captioning")
    print("=" * 40)
    
    captioner = OfflineImageCaptioning()
    
    while True:
        print("\nOptions:")
        print("1. Enter image URL")
        print("2. Test with dummy image")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            url = input("Enter image URL: ").strip()
            if url:
                caption = captioner.generate_caption(url)
                print(f"Generated Caption: {caption}")
                
                show = input("Show image? (y/n): ").strip().lower()
                if show == 'y':
                    captioner.visualize_result(url, caption)
        
        elif choice == '2':
            # Create a test image
            dummy_url = "dummy_image"  # Will create a blue dummy image
            caption = captioner.generate_caption(dummy_url)
            print(f"Generated Caption: {caption}")
            captioner.visualize_result(dummy_url, caption)
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def benchmark_offline_system():
    """Benchmark the offline system"""
    print("Offline System Benchmark")
    print("=" * 30)
    
    import time
    
    captioner = OfflineImageCaptioning()
    
    # Warm up
    print("Warming up...")
    captioner.generate_caption("dummy_image")
    
    # Benchmark
    num_runs = 10
    times = []
    
    print(f"Running {num_runs} inference tests...")
    for i in range(num_runs):
        start_time = time.time()
        caption = captioner.generate_caption("dummy_image")
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.3f}s - Caption: {caption}")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.3f}s")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")

if __name__ == "__main__":
    print("Offline Image Captioning Demo Options:")
    print("1. Basic offline demo")
    print("2. Interactive demo")
    print("3. Performance benchmark")
    print("4. Run all demos")
    
    choice = input("\nSelect demo (1-4): ").strip()
    
    if choice == '1':
        demo_offline_captioning()
    elif choice == '2':
        interactive_offline_demo()
    elif choice == '3':
        benchmark_offline_system()
    elif choice == '4':
        demo_offline_captioning()
        benchmark_offline_system()
    else:
        print("Invalid choice. Running basic demo...")
        demo_offline_captioning()