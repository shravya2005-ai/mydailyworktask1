import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json

class BetterImageCaptioning:
    """Better image captioning with more natural language generation"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V1').to(device)
        self.model.eval()
        
        # Load ImageNet class names
        self.class_names = self._load_imagenet_classes()
        
        # Better category mappings with more natural descriptions
        self.category_descriptions = {
            # Animals
            'dog': ['a dog', 'a canine', 'a pet dog'],
            'cat': ['a cat', 'a feline', 'a pet cat'],
            'bird': ['a bird', 'a flying bird', 'a feathered creature'],
            'horse': ['a horse', 'an equine', 'a majestic horse'],
            'cow': ['a cow', 'cattle', 'a farm animal'],
            'sheep': ['a sheep', 'a woolly sheep', 'livestock'],
            'elephant': ['an elephant', 'a large elephant', 'a majestic elephant'],
            'bear': ['a bear', 'a wild bear', 'a large bear'],
            'zebra': ['a zebra', 'a striped zebra', 'a wild zebra'],
            'giraffe': ['a giraffe', 'a tall giraffe', 'a long-necked giraffe'],
            
            # Vehicles
            'car': ['a car', 'an automobile', 'a vehicle'],
            'truck': ['a truck', 'a large truck', 'a commercial vehicle'],
            'bus': ['a bus', 'a public bus', 'a passenger bus'],
            'motorcycle': ['a motorcycle', 'a motorbike', 'a two-wheeler'],
            'bicycle': ['a bicycle', 'a bike', 'a two-wheeled bike'],
            'train': ['a train', 'a locomotive', 'a railway train'],
            'boat': ['a boat', 'a watercraft', 'a vessel'],
            'airplane': ['an airplane', 'an aircraft', 'a flying plane'],
            
            # People
            'person': ['a person', 'someone', 'an individual'],
            'man': ['a man', 'a male person', 'a gentleman'],
            'woman': ['a woman', 'a female person', 'a lady'],
            'child': ['a child', 'a young person', 'a kid'],
            
            # Nature
            'tree': ['a tree', 'trees', 'vegetation'],
            'flower': ['a flower', 'flowers', 'blooming flowers'],
            'grass': ['grass', 'green grass', 'a grassy area'],
            'mountain': ['a mountain', 'mountains', 'a mountainous landscape'],
            'beach': ['a beach', 'a sandy beach', 'a coastal scene'],
            'ocean': ['the ocean', 'water', 'a body of water'],
            'sky': ['the sky', 'a cloudy sky', 'an open sky'],
            
            # Objects
            'chair': ['a chair', 'seating', 'furniture'],
            'table': ['a table', 'a dining table', 'furniture'],
            'book': ['a book', 'books', 'reading material'],
            'bottle': ['a bottle', 'a container', 'a glass bottle'],
            'cup': ['a cup', 'a mug', 'a drinking vessel'],
            
            # Food
            'apple': ['an apple', 'fruit', 'fresh fruit'],
            'banana': ['a banana', 'tropical fruit', 'yellow fruit'],
            'pizza': ['pizza', 'food', 'a meal'],
            'cake': ['a cake', 'dessert', 'baked goods'],
            'bread': ['bread', 'baked goods', 'food']
        }
        
        # Scene templates for more natural captions
        self.scene_templates = [
            "This image shows {object} in {setting}",
            "The photo features {object} {action}",
            "A {adjective} scene with {object}",
            "An image of {object} {location}",
            "{object} can be seen {context}"
        ]
        
        self.settings = ['a natural setting', 'an outdoor environment', 'an indoor space', 'a urban area', 'a rural location']
        self.actions = ['resting', 'positioned', 'displayed', 'captured', 'shown']
        self.adjectives = ['beautiful', 'interesting', 'clear', 'detailed', 'colorful']
        self.locations = ['outdoors', 'indoors', 'in nature', 'in the scene', 'prominently']
        self.contexts = ['in the foreground', 'in the center', 'clearly', 'prominently', 'in detail']
    
    def _load_imagenet_classes(self):
        """Load ImageNet class names"""
        # This is a simplified mapping of common ImageNet classes to more natural names
        class_mapping = {
            # Dogs
            151: 'dog', 152: 'dog', 153: 'dog', 154: 'dog', 155: 'dog', 156: 'dog', 157: 'dog', 158: 'dog',
            159: 'dog', 160: 'dog', 161: 'dog', 162: 'dog', 163: 'dog', 164: 'dog', 165: 'dog', 166: 'dog',
            167: 'dog', 168: 'dog', 169: 'dog', 170: 'dog', 171: 'dog', 172: 'dog', 173: 'dog', 174: 'dog',
            175: 'dog', 176: 'dog', 177: 'dog', 178: 'dog', 179: 'dog', 180: 'dog', 181: 'dog', 182: 'dog',
            183: 'dog', 184: 'dog', 185: 'dog', 186: 'dog', 187: 'dog', 188: 'dog', 189: 'dog', 190: 'dog',
            191: 'dog', 192: 'dog', 193: 'dog', 194: 'dog', 195: 'dog', 196: 'dog', 197: 'dog', 198: 'dog',
            199: 'dog', 200: 'dog', 201: 'dog', 202: 'dog', 203: 'dog', 204: 'dog', 205: 'dog', 206: 'dog',
            207: 'dog', 208: 'dog', 209: 'dog', 210: 'dog', 211: 'dog', 212: 'dog', 213: 'dog', 214: 'dog',
            215: 'dog', 216: 'dog', 217: 'dog', 218: 'dog', 219: 'dog', 220: 'dog', 221: 'dog', 222: 'dog',
            223: 'dog', 224: 'dog', 225: 'dog', 226: 'dog', 227: 'dog', 228: 'dog', 229: 'dog', 230: 'dog',
            231: 'dog', 232: 'dog', 233: 'dog', 234: 'dog', 235: 'dog', 236: 'dog', 237: 'dog', 238: 'dog',
            239: 'dog', 240: 'dog', 241: 'dog', 242: 'dog', 243: 'dog', 244: 'dog', 245: 'dog', 246: 'dog',
            247: 'dog', 248: 'dog', 249: 'dog', 250: 'dog', 251: 'dog', 252: 'dog', 253: 'dog', 254: 'dog',
            255: 'dog', 256: 'dog', 257: 'dog', 258: 'dog', 259: 'dog', 260: 'dog', 261: 'dog', 262: 'dog',
            263: 'dog', 264: 'dog', 265: 'dog', 266: 'dog', 267: 'dog', 268: 'dog',
            
            # Cats
            281: 'cat', 282: 'cat', 283: 'cat', 284: 'cat', 285: 'cat',
            
            # Birds
            7: 'bird', 8: 'bird', 9: 'bird', 10: 'bird', 11: 'bird', 12: 'bird', 13: 'bird', 14: 'bird',
            15: 'bird', 16: 'bird', 17: 'bird', 18: 'bird', 19: 'bird', 20: 'bird', 21: 'bird', 22: 'bird',
            23: 'bird', 24: 'bird', 80: 'bird', 81: 'bird', 82: 'bird', 83: 'bird', 84: 'bird', 85: 'bird',
            86: 'bird', 87: 'bird', 88: 'bird', 89: 'bird', 90: 'bird', 91: 'bird', 92: 'bird', 93: 'bird',
            94: 'bird', 95: 'bird', 96: 'bird', 97: 'bird', 127: 'bird', 128: 'bird', 129: 'bird', 130: 'bird',
            131: 'bird', 132: 'bird', 133: 'bird', 134: 'bird', 135: 'bird', 136: 'bird', 137: 'bird', 138: 'bird',
            139: 'bird', 140: 'bird', 141: 'bird', 142: 'bird', 143: 'bird', 144: 'bird',
            
            # Vehicles
            407: 'car', 436: 'car', 511: 'car', 609: 'car', 627: 'car', 656: 'car', 705: 'car', 734: 'car',
            751: 'car', 817: 'car', 864: 'car', 867: 'car',
            408: 'truck', 569: 'truck', 675: 'truck', 717: 'truck', 757: 'truck', 867: 'truck',
            779: 'bus', 654: 'bus',
            671: 'motorcycle', 665: 'motorcycle',
            444: 'bicycle', 671: 'bicycle',
            820: 'train', 466: 'train',
            484: 'boat', 554: 'boat', 625: 'boat', 628: 'boat', 724: 'boat', 814: 'boat', 914: 'boat',
            404: 'airplane', 895: 'airplane',
            
            # Animals
            340: 'zebra', 291: 'elephant', 292: 'elephant', 293: 'elephant', 294: 'elephant',
            354: 'horse', 355: 'horse', 356: 'horse', 357: 'horse', 358: 'horse', 359: 'horse',
            345: 'cow', 346: 'cow', 347: 'cow', 348: 'cow',
            348: 'sheep', 349: 'sheep', 350: 'sheep', 351: 'sheep',
            295: 'bear', 296: 'bear', 297: 'bear', 298: 'bear',
            
            # Food items
            948: 'apple', 949: 'apple', 950: 'apple', 951: 'banana', 963: 'pizza',
            
            # Objects
            423: 'chair', 559: 'chair', 765: 'chair', 857: 'chair', 881: 'chair',
            532: 'table', 868: 'table',
            440: 'bottle', 898: 'bottle', 899: 'bottle', 901: 'bottle', 907: 'bottle',
            504: 'cup', 968: 'cup',
            
            # Default fallback
        }
        
        # Create reverse mapping with fallback
        classes = {}
        for i in range(1000):
            if i in class_mapping:
                classes[i] = class_mapping[i]
            else:
                classes[i] = 'object'  # Generic fallback
        
        return classes
    
    def load_image(self, image_path_or_url):
        """Load image from file path or URL with better error handling"""
        try:
            if image_path_or_url.startswith('http'):
                print(f"Loading image from URL: {image_path_or_url[:50]}...")
                response = requests.get(image_path_or_url, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                print(f"Successfully loaded image: {image.size}")
            else:
                print(f"Loading local image: {image_path_or_url}")
                image = Image.open(image_path_or_url).convert('RGB')
                print(f"Successfully loaded image: {image.size}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def predict_and_analyze(self, image):
        """Predict classes and analyze image content"""
        if image is None:
            return None, []
            
        # Show original image info
        print(f"Analyzing image of size: {image.size}")
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top 10 predictions for better analysis
        top10_prob, top10_catid = torch.topk(probabilities, 10)
        
        predictions = []
        print("\nTop 10 predictions:")
        for i in range(top10_prob.size(0)):
            class_idx = top10_catid[i].item()
            confidence = top10_prob[i].item()
            class_name = self.class_names.get(class_idx, 'unknown')
            predictions.append((class_name, confidence, class_idx))
            print(f"  {i+1}. {class_name} (ID: {class_idx}): {confidence:.4f}")
        
        return image, predictions
    
    def generate_natural_caption(self, predictions):
        """Generate more natural captions"""
        if not predictions:
            return "Unable to analyze image content"
        
        # Get the most confident prediction
        top_class, top_confidence, class_idx = predictions[0]
        
        print(f"\nGenerating caption for top prediction: {top_class} ({top_confidence:.4f})")
        
        # Generate caption based on confidence and class
        if top_confidence > 0.7:
            confidence_level = "clearly shows"
        elif top_confidence > 0.4:
            confidence_level = "appears to show"
        elif top_confidence > 0.2:
            confidence_level = "seems to contain"
        else:
            confidence_level = "might contain"
        
        # Get description variations
        if top_class in self.category_descriptions:
            descriptions = self.category_descriptions[top_class]
            main_desc = np.random.choice(descriptions)
        else:
            main_desc = f"a {top_class}" if top_class != 'object' else "various objects"
        
        # Check for secondary objects
        secondary_objects = []
        for class_name, confidence, _ in predictions[1:4]:  # Check next 3 predictions
            if confidence > 0.1 and class_name != top_class and class_name in self.category_descriptions:
                secondary_objects.append(class_name)
        
        # Build caption
        if secondary_objects:
            secondary_desc = ", ".join([self.category_descriptions[obj][0] for obj in secondary_objects[:2]])
            caption = f"This image {confidence_level} {main_desc}, with {secondary_desc} also visible"
        else:
            setting = np.random.choice(self.settings)
            caption = f"This image {confidence_level} {main_desc} in {setting}"
        
        # Add confidence info
        caption += f" (confidence: {top_confidence:.2f})"
        
        return caption
    
    def generate_caption(self, image_path_or_url):
        """Main caption generation function"""
        print(f"\n{'='*50}")
        print(f"PROCESSING: {image_path_or_url}")
        print(f"{'='*50}")
        
        # Load and analyze image
        image, predictions = self.predict_and_analyze(self.load_image(image_path_or_url))
        
        if predictions:
            caption = self.generate_natural_caption(predictions)
            print(f"\nFINAL CAPTION: {caption}")
            return caption
        else:
            return "Unable to process image"
    
    def visualize_result(self, image_path_or_url, caption, predictions=None):
        """Enhanced visualization"""
        image = self.load_image(image_path_or_url)
        if image is None:
            print("Could not load image for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image", fontsize=12)
        axes[0, 0].axis('off')
        
        # Caption
        axes[0, 1].text(0.1, 0.5, caption, fontsize=11, wrap=True, 
                       verticalalignment='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Generated Caption", fontsize=12)
        axes[0, 1].axis('off')
        
        # Top predictions bar chart
        if predictions:
            classes = [f"{pred[0]} ({pred[2]})" for pred in predictions[:8]]
            confidences = [pred[1] for pred in predictions[:8]]
            
            axes[1, 0].barh(classes, confidences, color='skyblue')
            axes[1, 0].set_xlabel('Confidence')
            axes[1, 0].set_title('Top 8 Predictions')
            axes[1, 0].invert_yaxis()
            
            # Confidence distribution
            all_confidences = [pred[1] for pred in predictions]
            axes[1, 1].hist(all_confidences, bins=10, alpha=0.7, color='lightgreen')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Confidence Distribution')
        
        plt.tight_layout()
        plt.show()

def demo_better_captioning():
    """Demo with better image analysis"""
    print("Better Image Captioning Demo")
    print("=" * 50)
    
    captioner = BetterImageCaptioning()
    
    # Test with a variety of images
    test_images = [
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",  # Dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",  # Cat
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500",  # Street/Car
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",  # Mountain
        "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=500",  # Food
    ]
    
    for i, img_url in enumerate(test_images):
        try:
            caption = captioner.generate_caption(img_url)
            
            show_viz = input(f"\nShow detailed visualization for image {i+1}? (y/n): ").strip().lower()
            if show_viz == 'y':
                image, predictions = captioner.predict_and_analyze(captioner.load_image(img_url))
                if predictions:
                    captioner.visualize_result(img_url, caption, predictions)
            
            input("Press Enter to continue to next image...")
            
        except Exception as e:
            print(f"Error processing image {i+1}: {e}")
    
    print("\nDemo completed!")

def interactive_better_demo():
    """Interactive demo with better analysis"""
    print("Interactive Better Image Captioning")
    print("=" * 50)
    
    captioner = BetterImageCaptioning()
    
    while True:
        print("\nOptions:")
        print("1. Enter image URL")
        print("2. Test with sample images")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            url = input("Enter image URL: ").strip()
            if url:
                try:
                    caption = captioner.generate_caption(url)
                    
                    show_viz = input("Show detailed analysis? (y/n): ").strip().lower()
                    if show_viz == 'y':
                        image, predictions = captioner.predict_and_analyze(captioner.load_image(url))
                        if predictions:
                            captioner.visualize_result(url, caption, predictions)
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == '2':
            demo_better_captioning()
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Better Image Captioning System")
    print("1. Run demo")
    print("2. Interactive mode")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == '1':
        demo_better_captioning()
    elif choice == '2':
        interactive_better_demo()
    else:
        print("Running demo...")
        demo_better_captioning()