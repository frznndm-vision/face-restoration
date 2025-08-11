# Face & Image Restoration Models (GFPGAN, PSFRGAN, DFDNet, DeblurGAN-v2, PMRF)

## 1. Project Overview

This repository presents **five state-of-the-art image restoration models**, each bundled with a ready-to-run Jupyter Notebook designed for Google Colab. These models cover a variety of restoration tasks—from face recovery to motion deblurring—offering a versatile toolkit for researchers and practitioners alike.

## 2. Models & References

- **GFPGAN** – *Towards Real-World Blind Face Restoration with Generative Facial Prior* (CVPR 2021). It leverages priors from a pretrained face GAN (e.g., StyleGAN) through spatial feature transforms, enabling fast and realistic face restoration and color enhancement in a single forward pass :contentReference[oaicite:0]{index=0}.

- **PSFRGAN** – *Progressive Semantic-Aware Style Transformation for Blind Face Restoration* (CVPR 2021). This model employs a multi-scale, semantic-guided progressive style transformation, using parsing maps to modulate features from coarse to fine, boosted by a semantic-aware style loss, and supported by a pretrained face parsing network :contentReference[oaicite:1]{index=1}.

- **DFDNet** – *Deep Face Deblurring Network* (ECCV 2020) uses a component dictionary (e.g., eyes, nose, mouth) extracted from high-quality images to restore facial details progressively. Ideal when facial components are identifiable and reference-quality images are available.

- **DeblurGAN-v2** – Presented in ICCV 2019, this GAN-based model with a Feature Pyramid Network (FPN) backbone (e.g., Inception-ResNet-v2 or MobileNet) offers efficient and high-quality motion deblurring for general images.

- **PMRF** – *Posterior-Mean Rectified Flow* (ICLR 2025). A recent flow-based approach optimized for minimizing the mean squared error (MSE), delivering highly accurate, photo-realistic image restoration.

## 3. Getting Started – Usage Guide

For each model’s notebook:

1. Open the corresponding `.ipynb` file in Google Colab.
2. Install dependencies (e.g., `pip install -r requirements.txt`).
3. Upload or link the required pretrained models and test images.
4. Run the cells sequentially to observe restoration results.

Feel free to customize input paths or tweak parameters to fit your use case.

## 4. Model Comparison

| Model           | Task                        | Strengths                                      | Notes / Considerations                             |
|----------------|-----------------------------|-----------------------------------------------|----------------------------------------------------|
| **GFPGAN**      | Face restoration + color    | Realistic, preserves identity, fast inference   | Focused on facial regions; background less treated |
| **PSFRGAN**     | Semantic face enhancement   | Multi-scale, parsing-guided detail restoration  | Requires parsing maps and more setup               |
| **DFDNet**      | Component-based face deblur | Component-level detail reproduction             | Needs component dictionary; best for aligned faces |
| **DeblurGAN-v2**| General motion deblurring   | High speed and quality, backbone flexibility    | General images; not specialized for faces          |
| **PMRF**        | Accurate MSE-optimized restoration | Photorealistic results                        | Newer model with possibly less available tooling   |

## 5. Recommended Use Cases

- **GFPGAN**: Ideal for restoring degraded face images or enhancing old photos while preserving identity.
- **PSFRGAN**: Best when you want semantically meaningful, refined detail restoration with control over face components.
- **DFDNet**: Use when component-level restoration is critical and reference data is available.
- **DeblurGAN-v2**: Optimal for general deblurring needs, such as motion-blurred photos.
- **PMRF**: Ideal when achieving the lowest MSE and highest pixel-level fidelity is the priority.

## 6. Citations

If you use any of these models in publications, please cite them accordingly:

```bibtex
@InProceedings{wang2021
