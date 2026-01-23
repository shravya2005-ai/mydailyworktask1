#!/usr/bin/env python3
"""
Production server runner for Image Captioning Web Application
"""

import os
import sys
from app import app

def main():
    """Run the Flask application"""
    print("=" * 60)
    print("🚀 Starting AI Image Captioning Web Application")
    print("=" * 60)
    
    # Check if required directories exist
    required_dirs = ['static', 'templates', 'static/uploads']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Missing directory: {directory}")
            return False
    
    print("✅ All required directories found")
    
    # Configuration
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Server will run on: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    try:
        print("\n🤖 Loading AI model...")
        # The model is loaded when the app starts
        print("✅ AI model loaded successfully!")
        
        print("\n🎯 Application ready!")
        print("📝 You can now:")
        print("   • Upload images for captioning")
        print("   • Use image URLs")
        print("   • Try sample images")
        print("\n" + "=" * 60)
        
        # Start the server
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)