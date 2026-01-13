# Plan: Making Diffusion Competitive with Pix2Pix

## Current Status

- **Pix2Pix baseline**: 26.65 dB PSNR (validated)
- **Diffusion approach**: Results pending (undertrained)

## Why Diffusion Underperforms

1. **`conv_in` is frozen** - conditioning channels initialized to zero, model can't learn to use input image
2. **Only 5K training steps** - StableSR/Marigold use 25K+
3. **Redundant losses** - `loss_noise` and `loss_latent_recon` are mathematically coupled
4. **Perceptual loss only 25% of steps** - inconsistent gradient signal
5. **20 inference steps** - too few for quality output
6. **VAE bottleneck** - destroys fine detail needed for restoration

---

## High-Priority Changes

### 1. Unfreeze `conv_in` (Critical)

**File**: `src/train.py`

After LoRA is applied (~line 189), add:

```python
# Unfreeze conv_in so model can learn to use conditioning channels
self.unet.conv_in.requires_grad_(True)
trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
logger.info(f"conv_in unfrozen. New trainable params: {trainable_params:,}")
```

**Why**: The 8-channel input has 4 channels for conditioning, but they're initialized to zero weights and frozen. The model literally cannot learn to use the input image through convolution.

---

### 2. Increase Training to 25K Steps

**File**: `configs/train_config.yaml`

```yaml
# Before
max_steps: 5000

# After
max_steps: 25000
save_steps: 5000
```

**Why**: Current training sees ~40K images. StableSR sees 800K+. The model needs more exposure to learn the mapping.

---

### 3. Compute Perceptual Loss Every Step

**File**: `configs/train_config.yaml`

```yaml
# Before
perceptual_every: 4

# After
perceptual_every: 1
```

**Why**: 75% of gradient updates currently have no perceptual guidance. This creates inconsistent optimization.

---

## Medium-Priority Changes

### 4. Switch to x0-Prediction

**File**: `configs/train_config.yaml`

```yaml
# Before
prediction_type: "epsilon"

# After
prediction_type: "sample"
```

**File**: `src/train.py` - simplify loss computation:

```python
# With sample prediction, the model directly predicts clean latents
# Remove loss_noise, just use loss_latent_recon + loss_perceptual
if prediction_type == "sample":
    loss_latent_recon = F.l1_loss(noise_pred, target_latents)  # noise_pred IS the clean prediction
    total_loss = lambda_latent * loss_latent_recon + lambda_perceptual * loss_perceptual
```

**Why**: For restoration tasks, predicting the answer directly is more natural than predicting noise. Also eliminates the redundant coupled losses.

---

### 5. Truncated Timestep Sampling

**File**: `src/train.py` (~line 520)

```python
# Before: uniform sampling across all noise levels
timesteps = torch.randint(
    0, self.noise_scheduler.config.num_train_timesteps,
    (batch_size,), device=self.device
).long()

# After: bias toward low noise (restoration is small perturbation)
max_timestep = self.noise_scheduler.config.num_train_timesteps // 2  # 500 instead of 1000
timesteps = torch.randint(0, max_timestep, (batch_size,), device=self.device).long()
```

**Why**: Print-to-camera is a subtle transformation, not generation from pure noise. Training on heavy noise levels wastes capacity.

---

### 6. Increase Inference Steps

**File**: `configs/train_config.yaml`

```yaml
# Before
num_inference_steps: 20

# After
num_inference_steps: 50
```

**Why**: More steps = better denoising quality. 20 is too aggressive for quality output.

---

## Lower-Priority / Experimental

### 7. Add Fidelity Control (StableSR-style)

**File**: `src/train.py` - add to `predict_single` method:

```python
def predict_single(self, input_image, num_steps=50, fidelity_weight=0.0):
    """
    Args:
        fidelity_weight: 0.0 = pure diffusion, 1.0 = pure VAE reconstruction
    """
    diffusion_output = self._diffusion_inference(input_image, num_steps)

    if fidelity_weight > 0:
        # Blend with VAE encode-decode (faithful but limited)
        input_latent = self.encode_images(input_image)
        vae_recon = self.decode_latents(input_latent)
        output = (1 - fidelity_weight) * diffusion_output + fidelity_weight * vae_recon
    else:
        output = diffusion_output

    return output
```

**Why**: Allows trading off between generative quality and faithful reconstruction at inference time.

---

### 8. ControlNet-Style Conditioning (Alternative to Channel Concat)

Instead of concatenating input latent with noisy target, use a ControlNet encoder to inject conditioning at multiple scales.

**Effort**: Significant refactor
**Benefit**: This is what StableSR/SUPIR do successfully

---

### 9. Pixel-Space Diffusion (Nuclear Option)

Skip VAE entirely, work directly on 512x512x3 images.

**Pros**: No information loss from VAE compression
**Cons**: Higher memory, slower training, needs architecture changes

---

## Implementation Order

| Priority | Change | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| 1 | Unfreeze `conv_in` | 1 line | High |
| 2 | 25K training steps | Config | High |
| 3 | `perceptual_every: 1` | Config | Medium |
| 4 | 50 inference steps | Config | Medium |
| 5 | x0-prediction | Config + code | Medium |
| 6 | Truncated timesteps | ~5 lines | Medium |
| 7 | Fidelity control | ~20 lines | Low |
| 8 | ControlNet conditioning | Major refactor | Unknown |

---

## Success Criteria

After implementing changes 1-6:

| Metric | Current Pix2Pix | Target Diffusion |
|--------|-----------------|------------------|
| PSNR | 26.65 dB | ≥26 dB |
| SSIM | 0.7454 | ≥0.74 |
| LPIPS | 0.2483 | ≤0.25 (should excel here) |

Diffusion should win on LPIPS (perceptual quality) even if slightly behind on PSNR/SSIM.

---

## References

- [StableSR (IJCV 2024)](https://github.com/IceClear/StableSR) - Controllable Feature Wrapping for fidelity control
- [CCSR](https://github.com/csslc/CCSR) - Content-consistent SR with fewer steps
- [Marigold](https://github.com/prs-eth/marigold) - Original Marigold conditioning approach
