"""
Visualization Utilities for SHAPE

Helper functions to create overlays and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple


def create_heatmap_overlay(rgb_img, saliency_map, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Create heatmap overlay on image.
    
    Args:
        rgb_img: RGB image (H, W, 3) in [0, 1]
        saliency_map: Saliency map (H, W)
        alpha: Blending factor
        colormap: OpenCV colormap
        
    Returns:
        Overlay image (H, W, 3)
    """
    # Normalize saliency to [0, 1]
    sal_min = saliency_map.min()
    sal_max = saliency_map.max()
    if sal_max > sal_min:
        saliency_norm = (saliency_map - sal_min) / (sal_max - sal_min)
    else:
        saliency_norm = np.zeros_like(saliency_map)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_norm), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Blend
    overlay = alpha * rgb_img + (1 - alpha) * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def visualize_saliency(rgb_img, saliency_map, title="SHAPE Explanation", save_path=None):
    """
    Create 3-panel visualization: original, saliency, overlay.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        saliency_map: Saliency map (H, W)
        title: Figure title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(rgb_img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Saliency
    im = axes[1].imshow(saliency_map, cmap='jet')
    axes[1].set_title('Saliency Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    overlay = create_heatmap_overlay(rgb_img, saliency_map)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_methods(rgb_img, saliency_maps: dict, save_path=None):
    """
    Compare multiple explanation methods side by side.
    
    Args:
        rgb_img: RGB image
        saliency_maps: Dict of {method_name: saliency_map}
        save_path: Path to save figure
    """
    n_methods = len(saliency_maps)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (method_name, sal_map) in enumerate(saliency_maps.items()):
        # Saliency map
        axes[0, idx].imshow(sal_map, cmap='jet')
        axes[0, idx].set_title(f'{method_name}\nSaliency', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Overlay
        overlay = create_heatmap_overlay(rgb_img, sal_map)
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'{method_name}\nOverlay', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_insertion_deletion_curves(results_dict: dict, save_path=None):
    """
    Plot insertion and deletion curves for multiple methods.
    
    Args:
        results_dict: Dict of {method_name: evaluation_results}
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for idx, (method_name, results) in enumerate(results_dict.items()):
        # Insertion
        ins_scores = results['insertion_scores']
        ins_auc = results['insertion_auc']
        x = np.linspace(0, 100, len(ins_scores))
        ax1.plot(x, ins_scores, '-', linewidth=2, color=colors[idx],
                label=f'{method_name} (AUC={ins_auc:.4f})')
        ax1.fill_between(x, 0, ins_scores, alpha=0.2, color=colors[idx])
        
        # Deletion
        del_scores = results['deletion_scores']
        del_auc = results['deletion_auc']
        ax2.plot(x, del_scores, '-', linewidth=2, color=colors[idx],
                label=f'{method_name} (AUC={del_auc:.4f})')
        ax2.fill_between(x, 0, del_scores, alpha=0.2, color=colors[idx])
    
    # Insertion plot
    ax1.set_xlabel('Percentage of Total Pixels Inserted', fontsize=12)
    ax1.set_ylabel('Prediction Probability', fontsize=12)
    ax1.set_title('Insertion Game (Higher AUC is better)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 1])
    
    # Deletion plot
    ax2.set_xlabel('Percentage of Total Pixels Removed', fontsize=12)
    ax2.set_ylabel('Prediction Probability', fontsize=12)
    ax2.set_title('Deletion Game (Lower AUC is better)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_adversarial_paradox(rgb_img, gradcam_map, shape_map, 
                                  gradcam_results, shape_results, save_path=None):
    """
    Recreate Figure 1 from the paper showing the SHAPE paradox.
    
    Args:
        rgb_img: RGB image
        gradcam_map: GradCAM saliency
        shape_map: SHAPE saliency
        gradcam_results: GradCAM evaluation results
        shape_results: SHAPE evaluation results
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Row 1: Saliency maps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(create_heatmap_overlay(rgb_img, gradcam_map))
    ax1.set_title('(a) GradCAM\n✓ Human Interpretable', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(create_heatmap_overlay(rgb_img, shape_map))
    ax2.set_title('(b) SHAPE (Ours)\n❌ NOT Interpretable', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Row 2: Deletion game
    ax3 = fig.add_subplot(gs[1, :])
    x = np.linspace(0, 100, len(gradcam_results['deletion_scores']))
    ax3.plot(x, gradcam_results['deletion_scores'], 'b-', linewidth=2, 
            label=f"GradCAM AUC = {gradcam_results['deletion_auc']:.4f}")
    ax3.fill_between(x, 0, gradcam_results['deletion_scores'], alpha=0.3, color='blue')
    ax3.plot(x, shape_results['deletion_scores'], 'r-', linewidth=2,
            label=f"SHAPE AUC = {shape_results['deletion_auc']:.4f}")
    ax3.fill_between(x, 0, shape_results['deletion_scores'], alpha=0.3, color='red')
    ax3.set_xlabel('Percentage of Total Pixels Removed', fontsize=11)
    ax3.set_ylabel('Prediction Probability', fontsize=11)
    ax3.set_title('(c) Deletion Game (Lower AUC is better)\nSHAPE Wins Objectively!', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Row 3: Insertion game
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(x, gradcam_results['insertion_scores'], 'b-', linewidth=2,
            label=f"GradCAM AUC = {gradcam_results['insertion_auc']:.4f}")
    ax4.fill_between(x, 0, gradcam_results['insertion_scores'], alpha=0.3, color='blue')
    ax4.plot(x, shape_results['insertion_scores'], 'r-', linewidth=2,
            label=f"SHAPE AUC = {shape_results['insertion_auc']:.4f}")
    ax4.fill_between(x, 0, shape_results['insertion_scores'], alpha=0.3, color='red')
    ax4.set_xlabel('Percentage of Total Pixels Inserted', fontsize=11)
    ax4.set_ylabel('Prediction Probability', fontsize=11)
    ax4.set_title('(d) Insertion Game (Higher AUC is better)\nSHAPE Wins Objectively!', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('The SHAPE Paradox: Objective Metrics ≠ Human Interpretability', 
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
