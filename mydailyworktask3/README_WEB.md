# AI Image Captioning Web Application

A modern, responsive web application for AI-powered image captioning using deep learning. Built with Flask and featuring a beautiful, user-friendly interface.

## 🌟 Features

### Core Functionality
- **AI-Powered Captioning**: Uses ResNet50 deep learning model for accurate image analysis
- **Multiple Input Methods**: Support for file uploads and image URLs
- **Real-time Processing**: Instant caption generation with confidence scores
- **Detailed Predictions**: Shows top 5 predictions with probability percentages

### User Interface
- **Modern Design**: Beautiful gradient backgrounds and smooth animations
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile devices
- **Drag & Drop**: Intuitive file upload with drag-and-drop support
- **Sample Images**: Pre-loaded sample images for quick testing
- **Visual Feedback**: Loading animations and error handling

### Technical Features
- **Secure File Handling**: Safe file upload with size and type validation
- **Image Processing**: Automatic image preprocessing and optimization
- **RESTful API**: Clean API endpoints for integration
- **Error Handling**: Comprehensive error messages and recovery

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Run the Application

```bash
python run_server.py
```

Or run directly with Flask:

```bash
python app.py
```

### 3. Open in Browser

Navigate to: `http://localhost:5000`

## 📁 Project Structure

```
image-captioning-web/
├── app.py                 # Main Flask application
├── run_server.py         # Production server runner
├── requirements_web.txt  # Python dependencies
├── README_WEB.md        # This file
├── templates/
│   └── index.html       # Main web interface
├── static/
│   ├── style.css        # Custom CSS styles
│   └── uploads/         # Uploaded images (auto-created)
└── models/              # AI model files (auto-downloaded)
```

## 🎨 User Interface

### Main Features

1. **Upload Section**
   - Drag & drop file upload
   - Click to browse files
   - File type and size validation
   - Visual feedback during upload

2. **URL Processing**
   - Direct image URL input
   - Sample image buttons
   - URL validation and error handling

3. **Results Display**
   - Original image preview
   - Generated caption in styled box
   - Prediction table with confidence bars
   - Animated result appearance

4. **Sample Images**
   - Pre-configured sample URLs
   - Quick testing options
   - Various image categories

### Design Elements

- **Gradient Backgrounds**: Modern blue-purple gradients
- **Glass Morphism**: Semi-transparent containers with blur effects
- **Smooth Animations**: Hover effects and transitions
- **Responsive Grid**: Adapts to different screen sizes
- **Icon Integration**: Font Awesome icons throughout

## 🔧 API Endpoints

### POST /upload
Upload an image file for captioning.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file

**Response:**
```json
{
  "success": true,
  "caption": "This image shows a dog",
  "predictions": [
    {
      "class": "dog",
      "confidence": 0.856,
      "index": 207
    }
  ],
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "filename": "uploaded_image.jpg"
}
```

### POST /url
Process an image from URL.

**Request:**
```json
{
  "url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "caption": "This image shows a cat",
  "predictions": [...],
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "url": "https://example.com/image.jpg"
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🛠️ Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

### Application Settings

- `MAX_CONTENT_LENGTH`: 16MB file size limit
- `UPLOAD_FOLDER`: static/uploads directory
- Model: ResNet50 with ImageNet weights

## 🎯 Usage Examples

### 1. File Upload
1. Click "Choose File" or drag image to upload area
2. Select an image file (JPG, PNG, etc.)
3. Wait for processing
4. View generated caption and predictions

### 2. URL Processing
1. Enter image URL in the input field
2. Click "Process URL" or press Enter
3. Wait for processing
4. View results

### 3. Sample Images
1. Click any sample image button
2. Automatic processing begins
3. View results for the sample image

## 🔍 Technical Details

### AI Model
- **Architecture**: ResNet50 Convolutional Neural Network
- **Training**: Pre-trained on ImageNet dataset
- **Classes**: 1000+ object categories
- **Input**: 224x224 RGB images
- **Output**: Class probabilities and natural language captions

### Image Processing
- **Preprocessing**: Resize, normalize, tensor conversion
- **Formats**: JPEG, PNG, GIF, BMP, TIFF
- **Size Limit**: 16MB maximum
- **Validation**: File type and size checking

### Caption Generation
- **Method**: Rule-based natural language generation
- **Confidence Levels**: High (>70%), Moderate (40-70%), Low (<40%)
- **Context**: Scene analysis and object relationships
- **Output**: Natural English sentences

## 🚀 Deployment

### Local Development
```bash
python run_server.py
```

### Production Deployment

#### Using Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt

COPY . .
EXPOSE 5000

CMD ["python", "run_server.py"]
```

### Cloud Deployment
- **Heroku**: Use Procfile with gunicorn
- **AWS**: Deploy on EC2 or Elastic Beanstalk
- **Google Cloud**: Use App Engine or Cloud Run
- **Azure**: Deploy on App Service

## 🔒 Security Features

- **File Validation**: Type and size checking
- **Secure Filenames**: Werkzeug secure_filename()
- **Input Sanitization**: URL and form validation
- **Error Handling**: Safe error messages
- **CORS Protection**: Same-origin policy

## 📱 Mobile Support

- **Responsive Design**: Bootstrap 5 grid system
- **Touch Friendly**: Large buttons and touch targets
- **Mobile Upload**: Camera and gallery access
- **Optimized Images**: Automatic image resizing
- **Fast Loading**: Optimized assets and caching

## 🎨 Customization

### Styling
- Edit `static/style.css` for custom styles
- Modify `templates/index.html` for layout changes
- Update color schemes in CSS variables
- Add custom animations and effects

### Functionality
- Extend `app.py` for new features
- Add new API endpoints
- Integrate additional AI models
- Implement user authentication

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure internet connection for model download
   - Check disk space for model files
   - Verify PyTorch installation

2. **File Upload Issues**
   - Check file size (max 16MB)
   - Verify file format (images only)
   - Ensure upload directory permissions

3. **URL Processing Errors**
   - Verify URL accessibility
   - Check image format at URL
   - Ensure stable internet connection

### Debug Mode
Enable debug mode for detailed error messages:
```bash
export DEBUG=true
python run_server.py
```

## 📊 Performance

### Benchmarks
- **Model Loading**: ~3-5 seconds (first time)
- **Image Processing**: ~0.5-2 seconds per image
- **Memory Usage**: ~500MB-1GB RAM
- **File Upload**: Up to 16MB images
- **Concurrent Users**: 10-50 (depending on hardware)

### Optimization Tips
- Use GPU for faster processing
- Implement image caching
- Add CDN for static assets
- Use load balancer for scaling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **torchvision**: For pre-trained models and transforms
- **Flask**: For the lightweight web framework
- **Bootstrap**: For responsive UI components
- **Font Awesome**: For beautiful icons
- **Unsplash**: For sample images

---

**Built with ❤️ using AI and modern web technologies**