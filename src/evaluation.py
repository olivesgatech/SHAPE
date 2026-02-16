"""
Evaluation Metrics for XAI Methods

Implements insertion and deletion games from:
Petsiuk et al. (2018) "RISE: Randomized Input Sampling for Explanation"
"""

import numpy as np
import torch
import cv2
from tqdm import tqdm


def insertion_game(model, image, saliency_map, steps=100, substrate='blur'):
    """
    Insertion Game: Progressively add pixels in order of importance.
    
    Higher AUC = Better explanation
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W)
        saliency_map: Saliency map (H, W)
        steps: Number of insertion steps
        substrate: 'blur' or 'black' for baseline
        
    Returns:
        scores: Prediction probability at each step
        auc: Area under curve
    """
    device = image.device
    _, C, H, W = image.shape
    
    # Create baseline (blurred or black image)
    if substrate == 'blur':
        baseline = torch.from_numpy(
            cv2.GaussianBlur(image[0].cpu().numpy().transpose(1, 2, 0), (11, 11), 5)
        ).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        baseline = torch.zeros_like(image)
    
    # Get pixel coordinates sorted by importance
    sal_flat = saliency_map.flatten()
    indices = np.argsort(sal_flat)[::-1]  # Descending order
    
    # Get predicted class
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()
    
    scores = []
    n_pixels = H * W
    pixels_per_step = n_pixels // steps
    
    current_image = baseline.clone()
    
    for step in range(steps + 1):
        # Calculate how many pixels to insert
        n_insert = min(step * pixels_per_step, n_pixels)
        
        # Create mask for inserted pixels
        mask = torch.zeros(H * W, device=device)
        if n_insert > 0:
            mask[indices[:n_insert]] = 1.0
        mask = mask.reshape(H, W)
        
        # Insert pixels: use original where mask=1, baseline where mask=0
        for c in range(C):
            current_image[0, c] = image[0, c] * mask + baseline[0, c] * (1 - mask)
        
        # Get prediction
        with torch.no_grad():
            output = model(current_image)
            prob = torch.softmax(output, dim=1)[0, pred_class].item()
            scores.append(prob)
    
    # Calculate AUC
    auc = np.trapz(scores, dx=1.0/steps)
    
    return np.array(scores), auc


def deletion_game(model, image, saliency_map, steps=100, substrate='blur'):
    """
    Deletion Game: Progressively remove pixels in order of importance.
    
    Lower AUC = Better explanation
    
    Args:
        model: Trained model
        image: Input image tensor (1, C, H, W)
        saliency_map: Saliency map (H, W)
        steps: Number of deletion steps
        substrate: 'blur' or 'black' for replacement
        
    Returns:
        scores: Prediction probability at each step
        auc: Area under curve
    """
    device = image.device
    _, C, H, W = image.shape
    
    # Create replacement (blurred or black)
    if substrate == 'blur':
        replacement = torch.from_numpy(
            cv2.GaussianBlur(image[0].cpu().numpy().transpose(1, 2, 0), (11, 11), 5)
        ).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        replacement = torch.zeros_like(image)
    
    # Get pixel coordinates sorted by importance
    sal_flat = saliency_map.flatten()
    indices = np.argsort(sal_flat)[::-1]  # Descending order
    
    # Get predicted class
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()
    
    scores = []
    n_pixels = H * W
    pixels_per_step = n_pixels // steps
    
    current_image = image.clone()
    
    for step in range(steps + 1):
        # Calculate how many pixels to delete
        n_delete = min(step * pixels_per_step, n_pixels)
        
        # Create mask for remaining pixels
        mask = torch.ones(H * W, device=device)
        if n_delete > 0:
            mask[indices[:n_delete]] = 0.0
        mask = mask.reshape(H, W)
        
        # Delete pixels: use replacement where mask=0, original where mask=1
        for c in range(C):
            current_image[0, c] = image[0, c] * mask + replacement[0, c] * (1 - mask)
        
        # Get prediction
        with torch.no_grad():
            output = model(current_image)
            prob = torch.softmax(output, dim=1)[0, pred_class].item()
            scores.append(prob)
    
    # Calculate AUC
    auc = np.trapz(scores, dx=1.0/steps)
    
    return np.array(scores), auc


def evaluate_explanation(model, image, saliency_map, steps=100):
    """
    Evaluate explanation using both insertion and deletion games.
    
    Args:
        model: Trained model
        image: Input image tensor
        saliency_map: Saliency map
        steps: Number of steps for evaluation
        
    Returns:
        Dictionary with scores and AUCs
    """
    ins_scores, ins_auc = insertion_game(model, image, saliency_map, steps)
    del_scores, del_auc = deletion_game(model, image, saliency_map, steps)
    
    return {
        'insertion_scores': ins_scores,
        'insertion_auc': ins_auc,
        'deletion_scores': del_scores,
        'deletion_auc': del_auc
    }


def pointing_game(saliency_map, bbox, tolerance=15):
    """
    Pointing Game: Check if maximum saliency is within object bbox.
    
    Args:
        saliency_map: Saliency map (H, W)
        bbox: Bounding box [x, y, w, h]
        tolerance: Pixel tolerance
        
    Returns:
        hit: 1 if max point is in bbox, 0 otherwise
    """
    # Find maximum point
    max_y, max_x = np.unravel_index(saliency_map.argmax(), saliency_map.shape)
    
    x, y, w, h = bbox
    
    # Check if point is within bbox + tolerance
    hit = (x - tolerance <= max_x <= x + w + tolerance and
           y - tolerance <= max_y <= y + h + tolerance)
    
    return int(hit)
