import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam.utils.image import preprocess_image as cam_preprocess, show_cam_on_image
from pathlib import Path
from tqdm import tqdm
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

from explanations_1 import SHAPE


def get_model_info():
    """Returns dictionary mapping model names to (model_fn, architecture_type)"""
    return {
        'resnet18': (
            lambda: models.resnet18(pretrained=True),
            'resnet'
        ),
        'resnet50': (
            lambda: models.resnet50(pretrained=True),
            'resnet'
        ),
        'vgg16': (
            lambda: models.vgg16(pretrained=True),
            'vgg'
        ),
        'densenet161': (
            lambda: models.densenet161(pretrained=True),
            'densenet'
        ),
        'mobilenet_v2': (
            lambda: models.mobilenet_v2(pretrained=True),
            'mobilenet'
        ),
        'mnasnet1_0': (
            lambda: models.mnasnet1_0(pretrained=True),
            'mnasnet'
        ),
    }


def load_progress(progress_file):
    """Load processing progress from file"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed': set(), 'failed': {}}


def save_progress(progress_file, progress):
    """Save processing progress to file"""
    progress_copy = progress.copy()
    progress_copy['completed'] = list(progress_copy['completed'])
    with open(progress_file, 'w') as f:
        json.dump(progress_copy, f, indent=2)


def get_image_paths(input_dir):
    """Get all image paths in val/class_index/image.jpg structure"""
    image_paths = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Traverse val/class_index/image.jpg structure
    for class_dir in sorted(input_path.iterdir()):
        if class_dir.is_dir():
            for img_file in sorted(class_dir.glob('*')):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(img_file)
    
    return image_paths


def create_output_path(input_path, input_dir, output_dir, model_name, p1_value, ext):
    """Create output path maintaining input directory structure"""
    # Get relative path from input_dir
    rel_path = input_path.relative_to(input_dir)
    
    # Create output path: output_dir/model_name/shape_p1/val/class_index/image.ext
    output_path = Path(output_dir) / model_name / f'shape_{p1_value}' / rel_path.parent / (rel_path.stem + ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    return output_path


def preprocess_image(image_path, device):
    """Preprocess image for model input - matches CAM processing exactly"""
    # Read and preprocess image exactly like CAM code
    rgb_img = cv2.imread(str(image_path), 1)[:, :, ::-1]
    if rgb_img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    rgb_img = np.float32(rgb_img) / 255
    
    # CRITICAL: Resize to 224×224 BEFORE preprocessing
    # This ensures input_tensor is always (1, 3, 224, 224)
    rgb_img_resized = cv2.resize(rgb_img, (224, 224))
    
    # Use the same preprocessing as CAM code on the RESIZED image
    input_tensor = cam_preprocess(
        rgb_img_resized,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)
    
    return input_tensor, rgb_img_resized


def create_overlay(rgb_img_resized, saliency_map):
    """Create overlay visualization - matches CAM code exactly"""
    # Normalize saliency map to [0, 1] if needed
    if saliency_map.max() > 1.0 or saliency_map.min() < 0.0:
        sal_min = saliency_map.min()
        sal_max = saliency_map.max()
        saliency_map = (saliency_map - sal_min) / (sal_max - sal_min + 1e-8)
    
    # Use the same overlay function as CAM code
    cam_image = show_cam_on_image(rgb_img_resized, saliency_map, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    return cam_image


def process_single_image(img_path, model, explainer, p1_value, model_name, 
                        input_dir, output_dir, device):
    """Process a single image with SHAPE - matches RISE processing flow exactly"""
    try:
        # Preprocess image exactly like CAM code
        input_tensor, rgb_img_resized = preprocess_image(img_path, device)
        
        # Validate tensor shape
        if input_tensor.shape != (1, 3, 224, 224):
            raise ValueError(f"Input tensor has wrong shape: {input_tensor.shape}, expected (1, 3, 224, 224)")
        
        # Get top-1 prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, dim=1)
            predicted_class = predicted_class.item()
        
        # Generate SHAPE explanation
        with torch.no_grad():
            saliency_maps = explainer(input_tensor)
            # Get saliency for predicted class
            saliency_map = saliency_maps[predicted_class].cpu().numpy()
        
        # Resize saliency map to 224x224 (matching CAM code)
        saliency_map_resized = cv2.resize(saliency_map, (224, 224))
        
        # Save numpy array
        npy_path = create_output_path(img_path, input_dir, output_dir, 
                                      model_name, p1_value, '.npy')
        np.save(npy_path, saliency_map_resized)
        
        # Create and save overlay image exactly like CAM code
        cam_image = create_overlay(rgb_img_resized, saliency_map_resized)
        
        jpg_path = create_output_path(img_path, input_dir, output_dir, 
                                      model_name, p1_value, '.jpg')
        cv2.imwrite(str(jpg_path), cam_image)
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error processing {img_path}: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg


def generate_or_load_masks(masks_dir, p1_value, N, s, input_size, device):
    """Generate masks for a specific p1 value or load if already exists"""
    mask_file = os.path.join(masks_dir, f'masks_p1_{p1_value}_N_{N}_s_{s}.npy')
    
    if os.path.exists(mask_file):
        print(f"Loading existing masks from {mask_file}")
        masks = np.load(mask_file)
        masks = torch.from_numpy(masks).float().to(device)
        return masks, N, p1_value
    
    print(f"Generating masks for p1={p1_value}, N={N}, s={s}")
    
    # Generate masks
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size
    
    grid = np.random.rand(N, s, s) < p1_value
    grid = grid.astype('float32')
    
    masks = np.empty((N, *input_size))
    
    from skimage.transform import resize
    for i in tqdm(range(N), desc=f'Generating masks for p1={p1_value}'):
        # Random shifts
        x = np.random.randint(0, int(cell_size[0]))
        y = np.random.randint(0, int(cell_size[1]))
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                               anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
    
    masks = masks.reshape(-1, 1, *input_size)
    
    # Save masks
    os.makedirs(masks_dir, exist_ok=True)
    np.save(mask_file, masks)
    print(f"Masks saved to {mask_file}")
    
    masks = torch.from_numpy(masks).float().to(device)
    return masks, N, p1_value


def process_model_with_p1(model_name, model_fn, image_paths, p1_value,
                          masks, N, input_dir, output_dir, device, 
                          gpu_batch, progress, progress_file):
    """Process all images for a single model with specific p1 value"""
    print(f"\n{'='*80}")
    print(f"Processing SHAPE: {model_name} with p1={p1_value}")
    print(f"{'='*80}")
    
    # Load model
    try:
        model = model_fn().to(device).eval()
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return
    
    # Create SHAPE explainer with pre-loaded masks
    explainer = SHAPE(model, input_size=(224, 224), gpu_batch=gpu_batch)
    explainer.masks = masks
    explainer.N = N
    explainer.p1 = p1_value
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_paths, desc=f"{model_name}/shape_{p1_value}"):
        # Create unique identifier for this task
        task_id = f"{model_name}/shape_{p1_value}/{img_path.relative_to(input_dir)}"
        
        # Check if already completed
        if task_id in progress['completed']:
            skip_count += 1
            continue
        
        # Skip if previously failed (don't retry errors on resume)
        if task_id in progress['failed']:
            skip_count += 1
            continue
        
        # Process image
        success, error = process_single_image(
            img_path, model, explainer, p1_value, model_name,
            input_dir, output_dir, device
        )
        
        if success:
            success_count += 1
            progress['completed'].add(task_id)
            
            # Save progress periodically (every 10 images)
            if success_count % 10 == 0:
                save_progress(progress_file, progress)
        else:
            fail_count += 1
            progress['failed'][task_id] = error
            print(f"\nFailed: {task_id}")
            print(f"Error: {error}")
    
    print(f"\n[{model_name}/shape_{p1_value}] Summary:")
    print(f"  Successful: {success_count}")
    print(f"  Skipped (already done): {skip_count}")
    print(f"  Failed: {fail_count}")
    
    # Save progress after each p1 value
    save_progress(progress_file, progress)
    
    # Clean up
    del model
    del explainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def get_args():
    parser = argparse.ArgumentParser(description='Batch SHAPE processing for multiple models')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Input directory with structure val/class_index/image.jpg')
    parser.add_argument('--output-dir', type=str, default='shape_outputs',
                       help='Output directory')
    parser.add_argument('--masks-dir', type=str, default='rise_masks',
                       help='Directory to save/load masks (can use same as RISE)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--N', type=int, default=4000,
                       help='Number of masks to generate')
    parser.add_argument('--s', type=int, default=8,
                       help='Grid size for mask generation')
    parser.add_argument('--p1-values', type=float, nargs='+', 
                       default=[0.1, 0.3, 0.5, 0.8],
                       help='List of p1 values for mask generation')
    parser.add_argument('--gpu-batch', type=int, default=400,
                       help='Batch size for GPU processing')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['resnet18', 'resnet50', 'vgg16', 'densenet161'],
                       help='Models to process')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from previous progress')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get image paths
    print(f"\nScanning input directory: {args.input_dir}")
    image_paths = get_image_paths(args.input_dir)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("No images found! Check your input directory structure.")
        return
    
    # Get models
    all_models = get_model_info()
    models_to_process = {k: v for k, v in all_models.items() if k in args.models}
    
    if len(models_to_process) == 0:
        print(f"No valid models selected. Available: {list(all_models.keys())}")
        return
    
    print(f"\nModels to process: {list(models_to_process.keys())}")
    print(f"p1 values: {args.p1_values}")
    print(f"Mask parameters: N={args.N}, s={args.s}")
    print(f"Total tasks: {len(models_to_process)} models × {len(args.p1_values)} p1 values × {len(image_paths)} images = {len(models_to_process) * len(args.p1_values) * len(image_paths)}")
    
    # Load or create progress tracker
    progress_file = os.path.join(args.output_dir, 'progress.json')
    if args.resume:
        progress = load_progress(progress_file)
        progress['completed'] = set(progress['completed'])
        # Convert failed dict keys to set for faster lookup
        progress['failed'] = {k: v for k, v in progress.get('failed', {}).items()}
        print(f"\nResuming from previous run.")
        print(f"  Already completed: {len(progress['completed'])} tasks")
        print(f"  Previously failed (will skip): {len(progress['failed'])} tasks")
    else:
        progress = {'completed': set(), 'failed': {}}
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate or load masks for each p1 value (can reuse RISE masks!)
    masks_dict = {}
    for p1_value in args.p1_values:
        masks, N, p1 = generate_or_load_masks(
            args.masks_dir, p1_value, args.N, args.s, (224, 224), device
        )
        masks_dict[p1_value] = (masks, N, p1)
    
    # Process each model with each p1 value
    for model_name, (model_fn, _) in models_to_process.items():
        for p1_value in args.p1_values:
            try:
                masks, N, p1 = masks_dict[p1_value]
                process_model_with_p1(
                    model_name, model_fn, image_paths, p1_value,
                    masks, N, args.input_dir, args.output_dir, device,
                    args.gpu_batch, progress, progress_file
                )
            except Exception as e:
                print(f"\nCritical error processing {model_name} with p1={p1_value}: {e}")
                print(traceback.format_exc())
                save_progress(progress_file, progress)
                continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total tasks completed: {len(progress['completed'])}")
    print(f"Total tasks failed: {len(progress['failed'])}")
    
    if progress['failed']:
        print("\nFailed tasks:")
        for task_id, error in list(progress['failed'].items())[:10]:
            print(f"  - {task_id}")
        if len(progress['failed']) > 10:
            print(f"  ... and {len(progress['failed']) - 10} more")
    
    # Save final progress
    save_progress(progress_file, progress)
    print(f"\nProgress saved to: {progress_file}")
    print("Done!")


if __name__ == '__main__':
    main()
