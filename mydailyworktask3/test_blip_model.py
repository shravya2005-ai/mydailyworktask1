#!/usr/bin/env python3
"""
Test script to verify BLIP model is working correctly
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

def test_blip_model():
    """Test the BLIP model with a sample image"""
    print("🚀 Testing BLIP Image Captioning Model")
    print("=" * 50)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Load model and processor
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")
        
        # Test with a sample image
        test_image_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500"
        print(f"\nTesting with image: {test_image_url}")
        
        # Load image
        response = requests.get(test_image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✅ Image loaded: {image.size}")
        
        # Generate caption
        print("Generating caption...")
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True,
                do_sample=False
            )
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"✅ Generated Caption: {caption}")
        
        # Test multiple captions
        print("\nGenerating multiple diverse captions...")
        captions = []
        for i in range(3):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    do_sample=True,
                    temperature=0.7 + (i * 0.3),
                    top_p=0.9,
                    early_stopping=True
                )
            
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
            print(f"  Caption {i+1}: {caption}")
        
        print("\n🎉 BLIP model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing BLIP model: {e}")
        return False

if __name__ == "__main__":
    success = test_blip_model()
    if success:
        print("\n✅ Model is ready for use in the web application!")
    else:
        print("\n❌ Model test failed. Please check your installation.")