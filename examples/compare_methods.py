"""
Compare SHAPE with GradCAM and RISE

This script reproduces the key findings from the paper by comparing
SHAPE against popular XAI methods.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import sys
sys.path.append('../src')

from shape import SHAPE, RISE
from evaluation import evaluate_explanation
from visualization import compare_methods, plot_insertion_deletion_curves, visualize_adversarial_paradox

# Optional: GradCAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except:
    print("pytorch-grad-cam not installed. GradCAM comparison will be skipped.")
    GRADCAM_AVAILABLE = False


def preprocess_image(image_path):
    """Preprocess image"""
    img = Image.open(image_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    # Also get RGB for visualization
    img_resized = img.resize((224, 224))
    rgb_img = np.array(img_resized) / 255.0
    
    return input_tensor, rgb_img


def get_gradcam_saliency(model, input_tensor, target_class):
    """Generate GradCAM explanation"""
    if not GRADCAM_AVAILABLE:
        return None
    
    # Get target layer (last conv layer for ResNet)
    target_layers = [model.layer4[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    return grayscale_cam


def main():
    # Configuration
    IMAGE_PATH = 'path/to/your/image.jpg'  # Change this!
    MASK_PATH = '../masks/comparison_masks.npy'
    OUTPUT_DIR = '../results'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading ResNet50...")
    model = models.resnet50(pretrained=True).to(device).eval()
    
    # Preprocess image
    print(f"Processing image: {IMAGE_PATH}")
    input_tensor, rgb_img = preprocess_image(IMAGE_PATH)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = nn.Softmax(dim=1)(output)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        predicted_class = top5_catid[0][0].item()
        print(f"\nPredicted class: {predicted_class}")
        print(f"Confidence: {top5_prob[0][0].item():.4f}")
    
    # Initialize explainers
    print("\nInitializing explainers...")
    
    # SHAPE
    shape_explainer = SHAPE(model, input_size=(224, 224), gpu_batch=100)
    try:
        shape_explainer.load_masks(MASK_PATH)
    except:
        print("Generating masks for SHAPE...")
        shape_explainer.generate_masks(N=2000, s=8, p1=0.5, savepath=MASK_PATH)
    
    # RISE (for comparison)
    rise_explainer = RISE(model, input_size=(224, 224), gpu_batch=100)
    rise_explainer.masks = shape_explainer.masks  # Reuse same masks
    rise_explainer.N = shape_explainer.N
    rise_explainer.p1 = shape_explainer.p1
    
    # Generate explanations
    print("\nGenerating explanations...")
    
    # SHAPE
    print("  - SHAPE...")
    with torch.no_grad():
        shape_saliency_maps = shape_explainer(input_tensor)
        shape_sal = shape_saliency_maps[predicted_class].cpu().numpy()
    
    # RISE
    print("  - RISE...")
    with torch.no_grad():
        rise_saliency_maps = rise_explainer(input_tensor)
        rise_sal = rise_saliency_maps[predicted_class].cpu().numpy()
    
    # GradCAM
    gradcam_sal = None
    if GRADCAM_AVAILABLE:
        print("  - GradCAM...")
        gradcam_sal = get_gradcam_saliency(model, input_tensor, predicted_class)
    
    # Collect saliency maps
    saliency_maps = {
        'SHAPE (Ours)': shape_sal,
        'RISE': rise_sal
    }
    if gradcam_sal is not None:
        saliency_maps['GradCAM'] = gradcam_sal
    
    # Visualize comparisons
    print("\nCreating visualizations...")
    compare_methods(rgb_img, saliency_maps, 
                   save_path=f'{OUTPUT_DIR}/comparison_saliency.png')
    
    # Evaluate with insertion/deletion games
    print("\nEvaluating with insertion/deletion games...")
    results = {}
    
    for method_name, sal_map in saliency_maps.items():
        print(f"  - {method_name}...")
        results[method_name] = evaluate_explanation(
            model, input_tensor, sal_map, steps=100
        )
        print(f"    Insertion AUC: {results[method_name]['insertion_auc']:.4f}")
        print(f"    Deletion AUC: {results[method_name]['deletion_auc']:.4f}")
    
    # Plot curves
    print("\nPlotting insertion/deletion curves...")
    plot_insertion_deletion_curves(results, 
                                  save_path=f'{OUTPUT_DIR}/comparison_curves.png')
    
    # Create paradox visualization (if GradCAM available)
    if gradcam_sal is not None:
        print("\nCreating paradox visualization...")
        visualize_adversarial_paradox(
            rgb_img, gradcam_sal, shape_sal,
            results['GradCAM'], results['SHAPE (Ours)'],
            save_path=f'{OUTPUT_DIR}/shape_paradox.png'
        )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nObjective Scores:")
    print("-" * 60)
    for method_name in results:
        ins_auc = results[method_name]['insertion_auc']
        del_auc = results[method_name]['deletion_auc']
        print(f"{method_name:15s}  Insertion: {ins_auc:.4f}  Deletion: {del_auc:.4f}")
    
    print("\nKey Finding:")
    print("-" * 60)
    if 'GradCAM' in results:
        shape_better_ins = results['SHAPE (Ours)']['insertion_auc'] > results['GradCAM']['insertion_auc']
        shape_better_del = results['SHAPE (Ours)']['deletion_auc'] < results['GradCAM']['deletion_auc']
        
        if shape_better_ins and shape_better_del:
            print("✓ SHAPE outperforms GradCAM on BOTH metrics")
            print("✗ But SHAPE is NOT human-interpretable!")
            print("\n→ This reveals the paradox: objective metrics can be fooled!")
        else:
            print("Results vary - try different images or parameters")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
