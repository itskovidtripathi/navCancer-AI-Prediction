import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import *

def load_model(model_path):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess the image for model input"""
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch, image

def predict_image(model, input_tensor):
    """Make prediction for a single image"""
    with torch.no_grad():
        output = model(input_tensor.to(DEVICE))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return predicted_class, confidence, probabilities

def plot_prediction(image, predicted_class, confidence, probabilities, class_names):
    """Plot the image with prediction results"""
    plt.figure(figsize=(12, 6))
    
    # Plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    
    # Plot the probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities.cpu().numpy())
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

def evaluate_single_image(model_path, image_path):
    """Evaluate a single image"""
    # Load model
    model = load_model(model_path)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(image_path)
    
    # Make prediction
    predicted_class, confidence, probabilities = predict_image(model, input_tensor)
    
    # Get class names
    class_names = list(CLASS_NAMES.values())
    
    # Plot results
    plot_prediction(original_image, predicted_class, confidence, probabilities, class_names)
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"{class_names[i]}: {prob:.2%}")

def evaluate_directory(model_path, directory_path):
    """Evaluate all images in a directory"""
    # Load model
    model = load_model(model_path)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No images found in {directory_path}")
        return
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        print(f"\nEvaluating {image_file}:")
        
        try:
            # Preprocess image
            input_tensor, original_image = preprocess_image(image_path)
            
            # Make prediction
            predicted_class, confidence, probabilities = predict_image(model, input_tensor)
            
            # Get class names
            class_names = list(CLASS_NAMES.values())
            
            # Plot results
            plot_prediction(original_image, predicted_class, confidence, probabilities, class_names)
            
            # Print results
            print(f"Predicted Class: {class_names[predicted_class]}")
            print(f"Confidence: {confidence:.2%}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate lung cancer detection model')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH,
                        help='Path to the trained model')
    parser.add_argument('--image', type=str, help='Path to a single image to evaluate')
    parser.add_argument('--directory', type=str, help='Path to directory containing images to evaluate')
    
    args = parser.parse_args()
    
    if args.image:
        evaluate_single_image(args.model, args.image)
    elif args.directory:
        evaluate_directory(args.model, args.directory)
    else:
        print("Please provide either --image or --directory argument") 