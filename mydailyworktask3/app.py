from flask import Flask, render_template, request, jsonify, url_for
import torch
from PIL import Image
import base64
import io
import os
import requests
from werkzeug.utils import secure_filename
import time
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ProperImageCaptioning:
    """Proper image captioning using pre-trained BLIP model"""
    
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print("Loading BLIP model for image captioning...")
        
        try:
            # Load the pre-trained BLIP model and processor
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model.to(self.device)
            self.model.eval()
            print("✅ BLIP model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading BLIP model: {e}")
            print("Falling back to basic captioning...")
            self.model = None
            self.processor = None 
    
    def load_image_from_file(self, file_path):
        """Load image from file path"""
        try:
            image = Image.open(file_path).convert('RGB')
            print(f"✅ Image loaded successfully: {image.size}")
            return image
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def load_image_from_url(self, url):
        """Load image from URL"""
        try:
            print(f"Loading image from URL: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
            print(f"✅ Image loaded successfully: {image.size}")
            return image
        except Exception as e:
            print(f"❌ Error loading image from URL: {e}")
            return None
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """Generate caption for an image using BLIP model"""
        if image is None:
            return "Error: Could not load image", 0.0
        
        if self.model is None or self.processor is None:
            return "Error: BLIP model not loaded properly", 0.0
        
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
            
            # Calculate a confidence score based on caption length and coherence
            confidence = min(0.95, max(0.7, len(caption.split()) / 10))
            
            return caption, confidence
            
        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return f"Error: Could not generate caption - {str(e)}", 0.0
    
    def generate_multiple_captions(self, image, num_captions=3, max_length=50):
        """Generate multiple diverse captions for an image"""
        if image is None:
            return ["Error: Could not load image"]
        
        if self.model is None or self.processor is None:
            return ["Error: BLIP model not loaded properly"]
        
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

# Initialize the captioning system
print("Initializing proper image captioning system...")
captioner = ProperImageCaptioning()
print("System ready!")

@app.route('/')
def index():
    """Main page"""
    # Add cache busting parameter
    cache_bust = int(datetime.now().timestamp())
    return render_template('index_maroon.html', cache_bust=cache_bust)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process image with BLIP model
            image = captioner.load_image_from_file(file_path)
            caption, confidence = captioner.generate_caption(image)
            
            # Generate multiple captions for variety
            multiple_captions = captioner.generate_multiple_captions(image, num_captions=3)
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'caption': caption,
                'confidence': confidence,
                'multiple_captions': multiple_captions,
                'image': f"data:image/jpeg;base64,{img_str}",
                'filename': filename
            })
    
    except Exception as e:
        print(f"Error in upload_image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/url', methods=['POST'])
def process_url():
    """Handle image URL processing"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Load image from URL
        image = captioner.load_image_from_url(url)
        if image is None:
            return jsonify({'error': 'Unable to load image from URL'}), 400
        
        # Process image with BLIP model
        caption, confidence = captioner.generate_caption(image)
        
        # Generate multiple captions for variety
        multiple_captions = captioner.generate_multiple_captions(image, num_captions=3)
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'caption': caption,
            'confidence': confidence,
            'multiple_captions': multiple_captions,
            'image': f"data:image/jpeg;base64,{img_str}",
            'url': url
        })
    
    except Exception as e:
        print(f"Error in process_url: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = captioner.model is not None and captioner.processor is not None
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model_status,
        'device': captioner.device
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)