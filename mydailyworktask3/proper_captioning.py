import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class ProperImageCaptioning:
    """A proper image captioning model using BLIP (Bootstrapping Language-Image Pre-training)"""
    
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print("Loading BLIP model for image captioning...")
        
        # Load the pre-trained BLIP model and processor
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)
        
        print("✅ Model loaded successfully!")
    
    def load_image(self, image_source):
        """Load image from file path or URL"""
        try:
            if isinstance(image_source, str):
                if image_source.startswith(('http://', 'https://')):
                    # Load from URL
                    print(f"Loading image from URL: {image_source}")
                    response = requests.get(image_source, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # Load from file path
                    print(f"Loading image from file: {image_source}")
                    image = Image.open(image_source).convert('RGB')
            else:
                # Assume it's already a PIL Image
                image = image_source.convert('RGB')
            
            print(f"✅ Image loaded successfully: {image.size}")
            return image
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def generate_caption(self, image_source, max_length=50, num_beams=5):
        """Generate caption for an image"""
        # Load the image
        image = self.load_image(image_source)
        if image is None:
            return "Error: Could not load image"
        
        print("🔍 Analyzing image and generating caption...")
        
        try:
            # Process the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode the generated caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            print(f"✅ Caption generated: {caption}")
            return caption
            
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return f"Error: Could not generate caption - {str(e)}"
    
    def generate_multiple_captions(self, image_source, num_captions=3, max_length=50):
        """Generate multiple diverse captions for an image"""
        image = self.load_image(image_source)
        if image is None:
            return ["Error: Could not load image"]
        
        print(f"🔍 Generating {num_captions} diverse captions...")
        
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            captions = []
            
            # Generate multiple captions with different parameters
            for i in range(num_captions):
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=5,
                        do_sample=True,
                        temperature=0.7 + (i * 0.3),  # Vary temperature for diversity
                        top_p=0.9,
                        early_stopping=True
                    )
                
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
                print(f"  Caption {i+1}: {caption}")
            
            return captions
            
        except Exception as e:
            print(f"❌ Error generating captions: {e}")
            return [f"Error: Could not generate caption - {str(e)}"]
    
    def caption_with_context(self, image_source, context_text="", max_length=50):
        """Generate caption with additional context"""
        image = self.load_image(image_source)
        if image is None:
            return "Error: Could not load image"
        
        print(f"🔍 Generating caption with context: '{context_text}'")
        
        try:
            # Process image and text together
            inputs = self.processor(image, context_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            print(f"✅ Contextual caption: {caption}")
            return caption
            
        except Exception as e:
            print(f"❌ Error generating contextual caption: {e}")
            return f"Error: Could not generate caption - {str(e)}"
    
    def visualize_caption(self, image_source, caption=None):
        """Display image with its caption"""
        image = self.load_image(image_source)
        if image is None:
            print("Cannot visualize - image loading failed")
            return
        
        if caption is None:
            caption = self.generate_caption(image_source)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Caption: {caption}", fontsize=14, pad=20, wrap=True)
        plt.tight_layout()
        plt.show()
    
    def batch_caption(self, image_sources, max_length=50):
        """Generate captions for multiple images"""
        results = []
        
        print(f"🔍 Processing {len(image_sources)} images...")
        
        for i, image_source in enumerate(image_sources):
            print(f"\nProcessing image {i+1}/{len(image_sources)}")
            caption = self.generate_caption(image_source, max_length)
            results.append({
                'image_source': image_source,
                'caption': caption
            })
        
        return results

def demo_proper_captioning():
    """Demonstrate the proper image captioning model"""
    print("🚀 Proper Image Captioning Demo")
    print("=" * 50)
    
    # Initialize the captioning model
    captioner = ProperImageCaptioning()
    
    # Test images
    test_images = [
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",  # Dog
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=500",  # Cat  
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500",  # Street
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",  # Mountain
        "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=500",  # Food
    ]
    
    for i, image_url in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"🖼️  TESTING IMAGE {i+1}")
        print(f"{'='*60}")
        
        try:
            # Generate single caption
            caption = captioner.generate_caption(image_url)
            print(f"📝 Single Caption: {caption}")
            
            # Generate multiple captions
            multiple_captions = captioner.generate_multiple_captions(image_url, num_captions=3)
            print(f"\n📝 Multiple Captions:")
            for j, cap in enumerate(multiple_captions):
                print(f"   {j+1}. {cap}")
            
            # Show visualization option
            show_viz = input(f"\n🖼️  Show image {i+1} with caption? (y/n): ").strip().lower()
            if show_viz == 'y':
                captioner.visualize_caption(image_url, caption)
            
            input("Press Enter to continue to next image...")
            
        except Exception as e:
            print(f"❌ Error processing image {i+1}: {e}")
    
    print(f"\n🎉 Demo completed!")

def interactive_proper_demo():
    """Interactive demo for proper image captioning"""
    print("🚀 Interactive Proper Image Captioning")
    print("=" * 50)
    
    captioner = ProperImageCaptioning()
    
    while True:
        print("\n📋 Options:")
        print("1. Caption image from URL")
        print("2. Caption local image file")
        print("3. Generate multiple captions")
        print("4. Caption with context")
        print("5. Batch process multiple images")
        print("6. Exit")
        
        choice = input("\n🔢 Choose option (1-6): ").strip()
        
        if choice == '1':
            url = input("🌐 Enter image URL: ").strip()
            if url:
                caption = captioner.generate_caption(url)
                print(f"📝 Caption: {caption}")
                
                show_viz = input("🖼️  Show image? (y/n): ").strip().lower()
                if show_viz == 'y':
                    captioner.visualize_caption(url, caption)
        
        elif choice == '2':
            file_path = input("📁 Enter image file path: ").strip()
            if file_path:
                caption = captioner.generate_caption(file_path)
                print(f"📝 Caption: {caption}")
                
                show_viz = input("🖼️  Show image? (y/n): ").strip().lower()
                if show_viz == 'y':
                    captioner.visualize_caption(file_path, caption)
        
        elif choice == '3':
            source = input("🌐 Enter image URL or file path: ").strip()
            if source:
                num_captions = int(input("🔢 Number of captions (default 3): ") or "3")
                captions = captioner.generate_multiple_captions(source, num_captions)
                print(f"\n📝 Generated {len(captions)} captions:")
                for i, cap in enumerate(captions):
                    print(f"   {i+1}. {cap}")
        
        elif choice == '4':
            source = input("🌐 Enter image URL or file path: ").strip()
            context = input("📝 Enter context (optional): ").strip()
            if source:
                caption = captioner.caption_with_context(source, context)
                print(f"📝 Contextual Caption: {caption}")
        
        elif choice == '5':
            print("📁 Enter image sources (URLs or file paths), one per line. Empty line to finish:")
            sources = []
            while True:
                source = input("   Image source: ").strip()
                if not source:
                    break
                sources.append(source)
            
            if sources:
                results = captioner.batch_caption(sources)
                print(f"\n📝 Batch Results:")
                for i, result in enumerate(results):
                    print(f"   {i+1}. {result['caption']}")
        
        elif choice == '6':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

def test_specific_images():
    """Test with specific challenging images"""
    print("🧪 Testing with Challenging Images")
    print("=" * 40)
    
    captioner = ProperImageCaptioning()
    
    # Challenging test cases
    challenging_images = [
        {
            'url': 'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=500',
            'description': 'Office/workspace scene'
        },
        {
            'url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500', 
            'description': 'Mountain landscape'
        },
        {
            'url': 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=500',
            'description': 'Cat portrait'
        }
    ]
    
    for test_case in challenging_images:
        print(f"\n🖼️  Testing: {test_case['description']}")
        print(f"🌐 URL: {test_case['url']}")
        
        caption = captioner.generate_caption(test_case['url'])
        print(f"📝 Generated Caption: {caption}")
        
        # Test multiple captions
        multiple = captioner.generate_multiple_captions(test_case['url'], 2)
        print(f"📝 Alternative Captions:")
        for i, alt_caption in enumerate(multiple):
            print(f"   {i+1}. {alt_caption}")
        
        print("-" * 40)

if __name__ == "__main__":
    print("🎯 Proper Image Captioning System")
    print("Choose your demo:")
    print("1. Full demo with sample images")
    print("2. Interactive demo")
    print("3. Test challenging images")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        demo_proper_captioning()
    elif choice == '2':
        interactive_proper_demo()
    elif choice == '3':
        test_specific_images()
    else:
        print("Running full demo...")
        demo_proper_captioning()