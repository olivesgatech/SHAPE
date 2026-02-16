# SHAPE Methodology

## Mathematical Formulation

### Core Concept: Necessity-Based Importance

SHAPE defines pixel importance based on **necessity** - how much the model's prediction drops when a pixel is removed.

### Formal Definition

For an image `I`, model `f`, and pixel location `λ`:

```
N_I,f(λ) = E_M[f(I) - f(I ⊙ M) | M(λ) = 0]
```

Where:
- `I`: Original input image
- `M`: Random binary mask
- `f(I)`: Model prediction probability on full image
- `f(I ⊙ M)`: Model prediction on masked image
- `M(λ) = 0`: Pixel at location λ is masked (removed)
- `⊙`: Element-wise multiplication

**Interpretation**: The importance of a pixel is the expected drop in prediction probability when that pixel is absent.

---

## Step-by-Step Algorithm

### 1. Generate Random Masks

```python
# Generate N random masks
for i in range(N):
    # Create small binary grid (s × s)
    grid = (random.rand(s, s) < p1).astype(float)
    
    # Upsample to input size with random shifts
    x, y = random_shift()
    mask[i] = resize(grid, upscale)[x:x+H, y:y+W]
```

**Parameters**:
- `N = 4000`: Number of masks
- `s = 8`: Grid size
- `p1 = 0.5`: Probability of pixel being visible

### 2. Get Baseline Prediction

```python
# CHANGE 1 from RISE: Get baseline on FULL image
baseline_pred = model(I)  # Shape: (1, num_classes)
```

### 3. Apply Masks and Get Predictions

```python
# Same as RISE: Apply masks and get predictions
masked_images = []
for mask in masks:
    masked_images.append(I * mask)

masked_preds = model(masked_images)  # Shape: (N, num_classes)
```

### 4. Compute Prediction Drops

```python
# CHANGE 2 from RISE: Compute DROP, not raw prediction
pred_drops = baseline_pred - masked_preds  # Shape: (N, num_classes)
```

**Key insight**: Positive values mean prediction decreased when pixels were masked.

### 5. Weight by Inverted Masks

```python
# CHANGE 3 from RISE: Use INVERTED masks
inverted_masks = 1.0 - masks  # Where mask=0 (masked) → 1, mask=1 (visible) → 0
```

**Why inverted?**: We want to credit pixels that were REMOVED (masked=0), not kept.

### 6. Aggregate Weighted Drops

```python
# CHANGE 4 from RISE: Aggregate drops weighted by inverted masks
saliency = matmul(pred_drops.T, inverted_masks.reshape(N, H*W))
saliency = saliency.reshape(num_classes, H, W)
```

### 7. Normalize

```python
# CHANGE 5 from RISE: Normalize by (1-p1) instead of p1
saliency = saliency / N / (1.0 - p1)
```

**Why (1-p1)?**: Because we're measuring impact of MASKED pixels (probability 1-p1).

---

## RISE vs SHAPE: Side-by-Side

| Aspect | RISE | SHAPE |
|--------|------|-------|
| **Philosophy** | Sufficiency | Necessity |
| **Question** | "Can these pixels produce prediction?" | "Are these pixels necessary?" |
| **Baseline** | None | `f(I)` on full image |
| **Prediction** | `f(I ⊙ M)` | `f(I) - f(I ⊙ M)` |
| **Mask weighting** | Original masks `M` | Inverted masks `1 - M` |
| **Credits** | Visible pixels | Masked pixels |
| **Normalization** | `/N/p1` | `/N/(1-p1)` |
| **Code changes** | - | 5 lines |

---

## Why SHAPE "Works" (Objectively)

### 1. Mathematically Sound

SHAPE directly implements the causal definition of necessity:
> "A feature is necessary if its absence causes the effect to not occur"

### 2. Model-Faithful

- Uses actual model predictions (no approximations)
- Directly measures prediction drops
- No gradient assumptions

### 3. Optimized for Metrics

**Deletion Game**: Remove important pixels → Prediction should drop
- SHAPE directly measures drops
- Highlights pixels whose removal causes maximum drop
- Perfect alignment with deletion metric!

**Insertion Game**: Add important pixels → Prediction should rise
- SHAPE highlights necessary pixels
- Adding them restores prediction
- High correlation with insertion metric!

---

## Why SHAPE Fails (Subjectively)

### 1. Non-Semantic Features

Neural networks rely on both:
- **Semantic features**: Object parts, textures
- **Non-semantic features**: Backgrounds, contexts, statistical patterns

SHAPE captures ALL necessary features, including non-semantic ones that humans don't understand.

### 2. Distributed Importance

Objects are recognized by combination of features. SHAPE might highlight:
- Background textures that help discrimination
- Edge patterns not visible to humans
- Statistical regularities in training data

### 3. Optimization Target

- **RISE**: Optimized for prediction (aligns with human perception)
- **SHAPE**: Optimized for prediction DROP (aligns with metrics, not humans)

---

## Monte Carlo Approximation

Full calculation requires summing over all possible masks:

```
N_I,f(λ) = Σ_m [f(I) - f(I⊙m)] · P[M=m|M(λ)=0]
```

Using Monte Carlo sampling:

```
N_I,f(λ) ≈ (1/N) Σ_{i=1}^N [f(I) - f(I⊙M_i)] · (1 - M_i(λ)) / (1 - p1)
```

Where:
- N random masks sampled
- Approximation improves with larger N
- N=4000 typically sufficient

---

## Implementation Details

### Mask Generation

```python
cell_size = ceil(input_size / s)
up_size = (s + 1) * cell_size

for i in range(N):
    # Binary grid
    grid = (rand(s, s) < p1).astype(float)
    
    # Random shifts
    x = randint(0, cell_size[0])
    y = randint(0, cell_size[1])
    
    # Upsample and crop
    mask[i] = resize(grid, up_size)[x:x+H, y:y+W]
```

### Batch Processing

```python
# Process masks in batches for memory efficiency
for i in range(0, N, gpu_batch):
    batch_masks = masks[i:i+gpu_batch]
    batch_preds = model(I * batch_masks)
    drops.append(baseline - batch_preds)
```

### GPU Optimization

- Keep masks on GPU
- Use torch operations (avoid loops)
- Batch predictions
- Typical: 400-800 batch size on modern GPUs

---

## Theoretical Properties

### Completeness

SHAPE satisfies:
```
Σ_λ N_I,f(λ) ≈ f(I) - E[f(I⊙M)]
```

Total importance ≈ Expected prediction drop

### Monotonicity

If pixel λ1 is more necessary than λ2:
```
N_I,f(λ1) > N_I,f(λ2)
```

### Linearity

For convex combination of images:
```
N_{αI1 + (1-α)I2, f} ≈ α·N_{I1,f} + (1-α)·N_{I2,f}
```

---

## Computational Complexity

### Time Complexity

- Mask generation: O(N · H · W)
- Forward passes: O(N · model_complexity)
- Aggregation: O(N · H · W · C)

Total: **O(N · H · W · (1 + C))**

Where:
- N: Number of masks (4000)
- H, W: Image height, width (224)
- C: Number of classes (1000)

### Space Complexity

- Masks: O(N · H · W) ≈ 4000 · 224 · 224 · 4 bytes ≈ 800 MB
- Predictions: O(N · C) ≈ 4000 · 1000 · 4 bytes ≈ 16 MB
- Saliency: O(H · W · C) ≈ 224 · 224 · 1000 · 4 bytes ≈ 200 MB

Total: **≈ 1 GB per image**

### Typical Runtime

On NVIDIA V100 GPU:
- Mask generation: ~30 seconds (once)
- Single image: ~5-10 seconds
- Batch of 100: ~2-3 minutes

---

## Comparison with Other Methods

### vs GradCAM

| Aspect | GradCAM | SHAPE |
|--------|---------|-------|
| Speed | Fast (single forward+backward) | Slow (N forwards) |
| Quality | Good | Better (objectively) |
| Interpretability | High | Low |
| Dependencies | Gradients | None |

### vs LIME

| Aspect | LIME | SHAPE |
|--------|------|-------|
| Method | Perturbation + Linear proxy | Perturbation + Direct |
| Accuracy | Approximate | Exact |
| Speed | Medium | Slow |
| Consistency | Variable | Stable |

### vs SHAP

| Aspect | SHAP | SHAPE |
|--------|------|-------|
| Basis | Game theory | Causal necessity |
| Guarantees | Additivity, Symmetry | Monotonicity |
| Speed | Very slow | Slow |
| Quality | Good | Better (objectively) |

---

## References

1. **Petsiuk et al. (2018)** - RISE: Randomized Input Sampling
2. **Fong & Vedaldi (2017)** - Meaningful Perturbation
3. **Watson et al. (2021)** - Necessity and Sufficiency
4. **Pearl (2009)** - Causality

---

## Key Takeaway

SHAPE is **adversarial** because it:
1. ✅ Satisfies all mathematical criteria for a good explanation
2. ✅ Outperforms others on objective metrics
3. ❌ Fails the ultimate test: human interpretability

This paradox reveals that **objective metrics can be gamed** and we need better evaluation methods that incorporate human judgment.
