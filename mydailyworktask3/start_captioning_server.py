#!/usr/bin/env python3
"""
Startup script for the image captioning web server
"""

import os
import sys
import subprocess
import time

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import torch
        import transformers
        from PIL import Image
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Run: python setup_proper_captioning.py")
        return False

def check_model():
    """Check if BLIP model can be loaded"""
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("✅ BLIP model components available")
        return True
    except ImportError as e:
        print(f"❌ BLIP model not available: {e}")
        return False

def start_server():
    """Start the Flask web server"""
    print("🚀 Starting Image Captioning Web Server...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return False
    
    if not check_model():
        return False
    
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Or on your network at: http://[your-ip]:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = start_server()
    if not success:
        print("\n💡 Troubleshooting:")
        print("1. Run setup: python setup_proper_captioning.py")
        print("2. Test model: python test_blip_model.py")
        print("3. Check requirements: pip install -r requirements_web.txt")
        sys.exit(1)