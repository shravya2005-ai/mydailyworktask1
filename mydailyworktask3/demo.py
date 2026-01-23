import torch
import matplotlib.pyplot as plt
from image_captioning import ImageCaptioningPipeline
import requests
from PIL import Image
from io import BytesIO

def demo_image_captioning():
    """Demonstrate image captioning with sample images"""
    print("Image Captioning AI Demo")
    print("=" * 50)
    
    # Initialize pipeline with transformer model
    print("Initializing model...")
    pipeline = ImageCaptioningPipeline(model_type='transformer')
    
    # Sample images for testing
    sample_images = [
        {
            "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500",
            "description": "Dog in nature"
        },
        {
            "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500", 
            "description": "City street"
        },
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",
            "description": "Mountain landscape"
        }
    ]
    
    # Process each image
    for i, img_info in enumerate(sample_images):
        print(f"\nProcessing Image {i+1}: {img_info['description']}")
        print("-" * 30)
        
        try:
            # Generate caption
            caption = pipeline.generate_caption(img_info['url'])
            print(f"Generated Caption: {caption}")
            
            # Display image with caption
            response = requests.get(img_info['url'])
            image = Image.open(BytesIO(response.content))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Generated: {caption}", fontsize=12, pad=20)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\nDemo completed!")

def compare_models():
    """Compare LSTM vs Transformer approaches"""
    print("Model Comparison Demo")
    print("=" * 30)
    
    # Sample image
    image_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500"
    
    # Test both models
    models = ['lstm', 'transformer']
    
    for model_type in models:
        print(f"\nTesting {model_type.upper()} model:")
        try:
            pipeline = ImageCaptioningPipeline(model_type=model_type)
            caption = pipeline.generate_caption(image_url)
            print(f"Caption: {caption}")
        except Exception as e:
            print(f"Error with {model_type}: {e}")

def interactive_demo():
    """Interactive demo where user can input image URLs"""
    print("Interactive Image Captioning Demo")
    print("=" * 40)
    
    pipeline = ImageCaptioningPipeline(model_type='transformer')
    
    while True:
        print("\nOptions:")
        print("1. Enter image URL")
        print("2. Use sample image")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            url = input("Enter image URL: ").strip()
            if url:
                try:
                    caption = pipeline.generate_caption(url)
                    print(f"Generated Caption: {caption}")
                    
                    # Ask if user wants to see the image
                    show = input("Show image? (y/n): ").strip().lower()
                    if show == 'y':
                        pipeline.visualize_result(url, caption)
                        
                except Exception as e:
                    print(f"Error: {e}")
        
        elif choice == '2':
            sample_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500"
            try:
                caption = pipeline.generate_caption(sample_url)
                print(f"Generated Caption: {caption}")
                pipeline.visualize_result(sample_url, caption)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

def benchmark_performance():
    """Benchmark model performance"""
    print("Performance Benchmark")
    print("=" * 25)
    
    import time
    
    pipeline = ImageCaptioningPipeline(model_type='transformer')
    sample_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?w=500"
    
    # Warm up
    print("Warming up model...")
    pipeline.generate_caption(sample_url)
    
    # Benchmark
    num_runs = 5
    times = []
    
    print(f"Running {num_runs} inference tests...")
    for i in range(num_runs):
        start_time = time.time()
        caption = pipeline.generate_caption(sample_url)
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.3f}s - Caption: {caption[:50]}...")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.3f}s")
    print(f"Min time: {min(times):.3f}s")
    print(f"Max time: {max(times):.3f}s")

if __name__ == "__main__":
    print("Image Captioning Demo Options:")
    print("1. Basic demo with sample images")
    print("2. Model comparison")
    print("3. Interactive demo")
    print("4. Performance benchmark")
    print("5. Run all demos")
    
    choice = input("\nSelect demo (1-5): ").strip()
    
    if choice == '1':
        demo_image_captioning()
    elif choice == '2':
        compare_models()
    elif choice == '3':
        interactive_demo()
    elif choice == '4':
        benchmark_performance()
    elif choice == '5':
        demo_image_captioning()
        compare_models()
        benchmark_performance()
    else:
        print("Invalid choice. Running basic demo...")
        demo_image_captioning()