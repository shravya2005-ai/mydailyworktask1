#!/usr/bin/env python3
"""
Setup script for proper image captioning with BLIP model
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    try:
        # Install web requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_model():
    """Test if the BLIP model can be loaded"""
    print("🧪 Testing BLIP model...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        
        print("Loading BLIP model (this may take a few minutes on first run)...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        print(f"✅ BLIP model loaded successfully on {device}!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading BLIP model: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'static/uploads',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Proper Image Captioning System")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Test model
    if not test_model():
        print("❌ Setup failed during model testing")
        print("💡 Try running: python test_blip_model.py")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the web application: python app.py")
    print("2. Open your browser to: http://localhost:5000")
    print("3. Upload an image or use a URL to test the captioning")
    print("\n💡 Tips:")
    print("- First model load may take a few minutes")
    print("- GPU will be used automatically if available")
    print("- Check test_blip_model.py for standalone testing")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)