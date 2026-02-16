"""
Basic SHAPE Usage Example

This script demonstrates how to generate SHAPE explanations for a single image.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../src')

from shape import SHAPE


def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Read image
    img = Image.open(image_path).convert('RGB')
    
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transforms
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    
    # Also get RGB numpy array for visualization
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    
    return input_batch, rgb_img


def visualize_saliency(rgb_img, saliency_map, save_path=None):
    """Create overlay visualization"""
    # Normalize saliency map to [0, 1]
    sal_min = saliency_map.min()
    sal_max = saliency_map.max()
    saliency_norm = (saliency_map - sal_min) / (sal_max - sal_min + 1e-8)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlay = 0.5 * rgb_img + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_norm, cmap='jet')
    axes[1].set_title('SHAPE Saliency Map')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def main():
    # Configuration
    IMAGE_PATH = 'path/to/your/image.jpg'  # Change this!
    MASK_PATH = '../masks/shape_masks.npy'
    OUTPUT_PATH = '../results/basic_example.png'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading ResNet50 model...")
    model = models.resnet50(pretrained=True).to(device).eval()
    
    # Initialize SHAPE explainer
    print("Initializing SHAPE explainer...")
    explainer = SHAPE(model, input_size=(224, 224), gpu_batch=100)
    
    # Generate or load masks
    try:
        print(f"Loading masks from {MASK_PATH}...")
        explainer.load_masks(MASK_PATH)
        print(f"Loaded {explainer.N} masks")
    except:
        print("Generating new masks...")
        explainer.generate_masks(N=4000, s=8, p1=0.5, savepath=MASK_PATH)
        print(f"Generated and saved {explainer.N} masks")
    
    # Preprocess image
    print(f"\nProcessing image: {IMAGE_PATH}")
    input_tensor, rgb_img = preprocess_image(IMAGE_PATH)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = nn.Softmax(dim=1)(output)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        print("\nTop 5 Predictions:")
        for i in range(5):
            print(f"  {i+1}. Class {top5_catid[0][i].item()}: {top5_prob[0][i].item():.4f}")
        
        predicted_class = top5_catid[0][0].item()
        print(f"\nPredicted class: {predicted_class}")
    
    # Generate SHAPE explanation
    print("\nGenerating SHAPE explanation...")
    with torch.no_grad():
        saliency_maps = explainer(input_tensor)
        saliency_map = saliency_maps[predicted_class].cpu().numpy()
    
    print(f"Saliency map shape: {saliency_map.shape}")
    print(f"Saliency range: [{saliency_map.min():.4f}, {saliency_map.max():.4f}]")
    
    # Visualize
    print("\nVisualizing results...")
    visualize_saliency(rgb_img, saliency_map, save_path=OUTPUT_PATH)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
