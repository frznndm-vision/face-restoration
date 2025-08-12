# Face Restoration Comparison

A collection of Google Colab notebooks implementing five face restoration/deblurring models:
GFPGAN, PSFRGAN, DFDNet, DeblurGAN-v2, and PMRF. Each notebook allows you to test restoration quality on the same input images and compare results seamlessly.

---

## 1. Models 

- **GFPGAN** — Generative Facial Prior for Blind Face Restoration
GFP-GAN uses a pretrained face GAN (Generative Facial Prior) to restore realistic and identity-preserving facial details from low-quality inputs. With novel channel-split spatial feature transform layers, it balances realness and fidelity in a single forward pass, avoiding the expensive per-image optimization required by GAN inversion. It simultaneously restores details and enhances colors, delivering superior results on both synthetic and real-world datasets.
- **PSFRGAN** — Progressive Semantic-Aware Style Transformation for Face Restoration
PSFR-GAN restores high-quality, realistic faces from low-quality images by combining pixel information with semantic guidance from parsing maps. Unlike traditional encoder–decoder designs, it applies multi-scale coarse-to-fine restoration via semantic-aware style transfer. A semantic-aware style loss improves texture details per facial region, and a pretrained parsing network ensures reliable semantic maps even from degraded inputs.
- **DFDNet** — Deep Face Dictionary Network for Face Restoration
DFDNet restores degraded face images by matching input facial components (eyes, nose, mouth) to a pretrained deep dictionary built from high-quality images using K-means clustering. Without needing a high-quality reference image of the same identity, it transfers fine details through a dictionary feature transfer (DFT) block that applies component-wise adaptive instance normalization (AdaIN) to align styles and uses a confidence score to adaptively fuse features. A multi-scale coarse-to-fine pipeline progressively refines the restoration for realistic outputs on real degraded faces.
- **DeblurGAN-v2** — End-to-End Generative Adversarial Network for Motion Deblurring
DeblurGAN-v2 introduces a novel relativistic conditional GAN with a double-scale discriminator designed for single-image motion deblurring. It is the first to incorporate a Feature Pyramid Network (FPN) into the generator, enabling flexible use of various backbones to balance performance and efficiency.
With heavy backbones like Inception-ResNet-v2, it achieves state-of-the-art deblurring quality. Using lightweight backbones such as MobileNet variants, it runs 10 to 100 times faster than competitors while maintaining comparable results—making it suitable for real-time video deblurring. The architecture is also adaptable to other image restoration tasks.
- **PMRF** — PMRF (Posterior-Mean Rectified Flow)
Photo-realistic Image Restoration via Optimal Transport
PMRF proposes a novel approach to image restoration that aims to minimize distortion (MSE) while perfectly preserving perceptual quality, meaning the restored images follow the exact distribution of ground-truth images. Unlike typical methods balancing distortion and perceptual losses, PMRF constructs an optimal estimator by first predicting the posterior mean (MMSE estimate) and then transforming it using a rectified flow model that approximates an optimal transport map. This two-step process better aligns restored images with true data distribution, leading to improved photo-realistic quality.
---

## 2. Comparison Table

| Model         | Strengths                                     | Limitations                          |
|---------------|-----------------------------------------------|--------------------------------------|
| GFPGAN        | Realistic, identity-preserving, color-rich results; efficient single-pass restoration; superior performance over earlier methods.     | Struggles with extremely degraded inputs and large pose variations; may introduce color bias when input lacks color information. |
| PSFRGAN       | Semantic-detail enhancement using parsing maps | Restoration quality depends on the accuracy of parsing maps     |
| DFDNet        | Limited to restoring facial components present in the dictionary, so it depends on the quality and coverage of the dictionary.    | Interpretable, component-based restoration that does not require identity-specific reference images. |
| DeblurGAN-v2  | Flexible and efficient deblurring with state-of-the-art quality, scalable from real-time lightweight models to high-performance backbones.     | Higher accuracy requires heavy backbones, which increase computational cost and reduce speed. |
| PMRF          | Achieves superior photo-realistic restoration by optimally balancing distortion minimization and perfect perceptual quality using flow-based optimal transport.        | Requires complex training and higher computational resources due to the flow-based rectification step.      |

---

## 3. Usage Instructions (Google Colab)

Each model has its own Colab notebook (e.g., `GFPGAN_colab.ipynb`, `PSFRGAN_colab.ipynb`, etc.) containing:

1. **Setup**: Installs dependencies and clones the corresponding repository.
2. **Model Loading**: Downloads and initializes pretrained weights.
3. **Inference**: Runs restoration on sample or user-uploaded images.
4. **Outputs**: Saves results under `outputs/<model>/`.

**To use:**

1. Open the desired notebook in Colab.
2. Select **GPU** as runtime type.
3. Upload input images into the designated folder.
4. Run all cells to generate outputs (found in `outputs/<model>/`).

---

## 4. Sample Results Table


## Sample Restoration Comparisons

| Original Image | GFPGAN Output | PSFRGAN Output | DFDNet Output | DeblurGAN-v2 Output | PMRF Output |
|----------------|--------------|----------------|---------------|---------------------|-------------|
| ![1](https://github.com/user-attachments/assets/3e87edb6-bf80-4938-84dd-236d347d710f) | ![GFPGAN1](https://github.com/user-attachments/assets/f69b4b46-b382-4877-81ed-4ae4219ee0b8) | ![PSFR1](path/to/psfr1.png) | ![DFD1](path/to/dfd1.png) | ![Deblur1](path/to/deblur1.png) | ![PMRF1](path/to/pmrf1.png) |
| ![2](https://github.com/user-attachments/assets/cbae896d-fa8c-4e26-a8f6-c76b2dbd5567) |![2](https://github.com/user-attachments/assets/060fd1de-9829-4fdb-886a-59f56415ea96)
  | ![PSFR2](path/to/psfr2.png) | ![DFD2](path/to/dfd2.png) | ![Deblur2](path/to/deblur2.png) | ![PMRF2](path/to/pmrf2.png) |
| <img width="114" height="142" alt="3" src="https://github.com/user-attachments/assets/6670c5d6-af50-4534-9378-b3a7f97243f6" /> | ![GFPGAN3](path/to/gfpgan3.png) | ![PSFR3](path/to/psfr3.png) | ![DFD3](path/to/dfd3.png) | ![Deblur3](path/to/deblur3.png) | ![PMRF3](path/to/pmrf3.png) |
| ![4](https://github.com/user-attachments/assets/498c36b8-9f2c-4340-9285-625b44cbd234) | ![GFPGAN4](path/to/gfpgan4.png) | ![PSFR4](path/to/psfr4.png) | ![DFD4](path/to/dfd4.png) | ![Deblur4](path/to/deblur4.png) | ![PMRF4](path/to/pmrf4.png) |
| <img width="114" height="142" alt="7" src="https://github.com/user-attachments/assets/9333c797-2517-440b-8deb-2af083d583ef" /> | ![GFPGAN7](path/to/gfpgan7.png) | ![PSFR7](path/to/psfr7.png) | ![DFD7](path/to/dfd7.png) | ![Deblur7](path/to/deblur7.png) | ![PMRF7](path/to/pmrf7.png) |

## References

- **GFP-GAN:**  
  Wang et al., *GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior*, 2021.  
  [arXiv](https://arxiv.org/abs/2101.04061) | [GitHub](https://github.com/TencentARC/GFPGAN)

- **PSFR-GAN:**  
  Zhang et al., *PSFR-GAN: Progressive Semantic-Aware Style Transformation for Blind Face Restoration*, 2020.  
  [arXiv](https://arxiv.org/abs/2009.08709) | [GitHub](https://github.com/chaofengc/PSFR-GAN)

- **DFDNet:**  
  Li et al., *Blind Face Restoration via Deep Multi-scale Component Dictionaries*, 2020.  
  [arXiv](https://arxiv.org/abs/2008.00418) | [GitHub](https://github.com/csjliang/DFDNet)

- **DeblurGAN-v2:**  
  Kupyn et al., *DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better*, 2019.  
  [arXiv](https://arxiv.org/abs/1908.03826) | [GitHub](https://github.com/KupynOrest/DeblurGANv2)

- **PMRF:**  
  Saharia et al., *Posterior-Mean Rectified Flow for Image Restoration*, 2021.  
  [arXiv](https://arxiv.org/abs/2112.03563) | [GitHub](https://github.com/saharia/pmrf)


