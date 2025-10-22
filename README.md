# Real-World Blind Face Restoration Comparison

A collection of **five Google Colab notebooks** (GFPGAN, CodeFormer, DFDNet, DifFace, PMRF) **and one local implementation** (**InstantRestore**).  
The Colab notebooks let you easily test restoration quality on shared input images and compare results interactively, while **InstantRestore** runs locally via Python.


---

## 1. Models

* **GFPGAN** — Generative Facial Prior for Blind Face Restoration
  GFP-GAN uses a pretrained face GAN (Generative Facial Prior) to restore realistic and identity-preserving facial details from low-quality inputs. With novel channel-split spatial feature transform layers, it balances realness and fidelity in a single forward pass, avoiding the expensive per-image optimization required by GAN inversion. It simultaneously restores details and enhances colors, delivering superior results on both synthetic and real-world datasets.

* **CODEFORMER** — Transformer-Based Codebook Prior for Blind Face Restoration
  CodeFormer restores high-quality, realistic faces from degraded inputs by reframing restoration as a discrete code prediction problem. A learned codebook prior in a compact latent space reduces ambiguity and provides rich “visual atoms” for reconstruction. Its Transformer-based prediction network models global composition and context, enabling faithful recovery even from severely degraded images. A controllable feature transformation module allows flexible adjustment between fidelity and quality, making the method robust across diverse degradation types and achieving state-of-the-art performance on synthetic and real-world datasets.

* **DFDNet** — Deep Multi-scale Component Dictionary for Blind Face Restoration
  DFDNet restores degraded face images by matching input facial components (eyes, nose, mouth) to a pretrained deep dictionary built from high-quality images using K-means clustering. Without needing a high-quality reference image of the same identity, it transfers fine details through a dictionary feature transfer (DFT) block that applies component-wise adaptive instance normalization (AdaIN) to align styles and uses a confidence score to adaptively fuse features. A multi-scale coarse-to-fine pipeline progressively refines the restoration for realistic outputs on real degraded faces.

* **DifFace** — Blind Face Restoration with Diffused Error Contraction
  DifFace restores realistic, identity-preserving faces from degraded inputs using a Diffused Error Contraction (DEC) framework. Instead of direct restoration, it applies a diffusion process guided by a pretrained generative prior to iteratively contract errors toward the high-quality face manifold. This design improves robustness to unknown degradations, preserves identity details, and delivers high-fidelity results on both synthetic and real-world datasets.

* **PMRF** — Posterior-Mean Rectified Flow
  Photo-realistic Image Restoration via Optimal Transport. PMRF proposes a novel approach to image restoration that aims to minimize distortion (MSE) while perfectly preserving perceptual quality, meaning the restored images follow the exact distribution of ground-truth images. Unlike typical methods balancing distortion and perceptual losses, PMRF constructs an optimal estimator by first predicting the posterior mean (MMSE estimate) and then transforming it using a rectified flow model that approximates an optimal transport map. This two-step process better aligns restored images with true data distribution, leading to improved photo-realistic quality.

* **InstantRestore — Single-Step Personalized Face Restoration (Snap Research, 2024)**
  InstantRestore is a **next-generation, one-step diffusion-based face restoration model** developed by **Snap Research**. It integrates **shared-image attention** to utilize one or multiple high-quality reference images of the same identity, achieving both realism and identity preservation in a single forward pass.

  * Uses a pretrained diffusion backbone that learns direct mapping from degraded to restored faces with shared latent space conditioning.
  * Incorporates **cross-image attention** to match low-quality patches with high-quality identity-specific patches from the reference set.
  * Demonstrates strong performance on low-quality real-world and surveillance-like data, outperforming existing restoration methods in both visual quality and speed.
  * GitHub: [InstantRestore (Snap Research)](https://github.com/snap-research/InstantRestore) | [Project Page](https://snap-research.github.io/InstantRestore/)

---

## 2. Comparison Table

| Model          | Strengths                                                                                                                           | Limitations                                                                        |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| GFPGAN         | Realistic, identity-preserving, color-rich results; efficient single-pass restoration.                                              | Struggles with extreme degradation and pose variation; may introduce color bias.   |
| CODEFORMER     | Codebook prior with Transformer-based global context; adjustable fidelity-quality trade-off.                                        | May alter identity under extreme degradation; computationally heavier.             |
| DFDNet         | Interpretable component-based restoration; no need for identity reference.                                                          | Dependent on dictionary quality and component coverage.                            |
| DifFace        | Robust to unknown degradations; strong identity preservation via diffusion.                                                         | Slower inference; dependent on generative prior quality.                           |
| PMRF           | Balances perceptual quality and low distortion through flow-based optimal transport.                                                | Requires high computational resources and complex training.                        |
| InstantRestore | Single-step diffusion-based model; very fast; preserves identity through shared-image attention using high-quality references; demonstrates strong performance on real-world and surveillance-like data (Snap Research, 2024) | Requires one or more reference images per identity; may underperform without them. |

---

## 3. Usage Instructions (Google Colab)

Colab notebooks are provided for: GFPGAN, CodeFormer, DFDNet, DifFace, PMRF.
Note: InstantRestore does not have a Colab notebook. Use the local workflow in 3.1.:

1. **Setup:** Installs dependencies and clones the corresponding repository.
2. **Model Loading:** Downloads and initializes pretrained weights.
3. **Inference:** Runs restoration on sample or user-uploaded images.
4. **Outputs:** Saves results under `outputs/<model>/`.

**To use:**

1. Open the desired notebook in Colab.
2. Select **GPU** as runtime type.
3. Upload input images into the designated folder.
4. Run all cells to generate outputs (found in `outputs/<model>/`).

#3.1 Local Usage (Non‑Colab)


## Get the code
```bash
# Clone the upstream repo (reference)
git clone https://github.com/snap-research/InstantRestore.git
cd InstantRestore

# OR clone your own fork
# git clone https://github.com/<your-username>/<your-repo>.git
# cd <your-repo>
```
> Make sure you place all checkpoints inside this repository folder (e.g., `./checkpoints`).

## Prerequisites
- Linux + NVIDIA GPU (tested: **GTX 1080 Ti** ~11GB)
- Conda + Python 3.10

## Fast Installation
```bash
conda create -n instantrestore python=3.10 -y
conda activate instantrestore

# PyTorch compatible with 10xx series (CUDA 11.8)
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2

# Dependencies
pip install numpy==1.26.4 pillow natsort pyrallis einops safetensors opencv-python tqdm
pip install diffusers==0.27.2 transformers==4.41.2 huggingface_hub==0.23.4 peft==0.10.0
pip install insightface==0.7.3 onnxruntime==1.17.3
```

## Memory Settings
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export INSIGHTFACE_ONNX_PROVIDERS=CPUExecutionProvider
```

## Models
- Place the InstantRestore checkpoint at:
```
./checkpoints/final_model_ckpt.pt
```

## Run

```bash
python face_replace/inference/test.py   --ckpt ./checkpoints/final_model_ckpt.pt   --src_dir ./samples_in   --save_dir ./IR_results_selfref   --max_refs 1 --skip_existing
```

## 4. Results Table
### Visual Comparison

![output2](images/output2.png)

---

## 5. Experimental Evaluation on SCFace Dataset

To further analyze the real-world behavior of face restoration models, an additional evaluation was conducted using the **[SCFace dataset](https://www.scface.org/)**, which contains real surveillance camera images captured under varying distances and qualities.
Two state-of-the-art restoration methods — **[InstantRestore (Snap Research)](https://github.com/snap-research/InstantRestore)** and **[GFPGAN](https://github.com/TencentARC/GFPGAN)** — were applied to low-quality facial images from SCFace.

### Evaluation Method

* **Identity Similarity Measurement:**
  Identity similarity between restored and reference (high-quality) faces was computed using **[InsightFace](https://github.com/deepinsight/insightface)** with the **Antelope** and **Buffalo** recognition models.
* **Metric:** Cosine similarity scores were used to assess how much the restored faces preserved the identity of the original subject.
* **Comparison:** Results were compared **before** and **after** restoration to quantify whether enhancement improves identity similarity.

### Observations

Contrary to initial expectations, the quantitative results showed that **identity similarity scores decreased after applying the restoration models**, despite the visual improvements in sharpness and perceptual quality.
This indicates that while modern restoration models effectively enhance image clarity and realism, they can **alter facial features in ways that reduce feature-space consistency** with the original identity — especially for faces captured in challenging, low-resolution surveillance settings.

---

## 6. Identity Similarity Results (Measured by InsightFace)

| Method           | Cosine Similarity (↑) — **Buffalo** | Cosine Similarity (↑) — **Antelope** |
|------------------|----------------------------------------|----------------------------------------|
| Original (Before Restoration) | 0.40 | 
| GFPGAN (Restored) | 0.35 | 0.33 | 
| InstantRestore (Restored) | 0.34 | 0.31 | 


---
## 5. Future Work

To further improve the performance and applicability of face restoration models, the following extensions are planned:

  - **Custom Dataset Creation:** Build a curated dataset with varied types of real-world distortions(e.g., blur,noise,compression artifacts,occlusions,low-light conditions). This dataset will simulate diverse and realistic degradation scenarios, aiming to improve generalization.
  - **Data Augmentation Strategies:** Apply advanced data augmentation techniques(e.g., random occlusion, synthetic aging, facial misalignment) to enrich the trainig data and enhance model robustness.
  - **Model Fine-Tuning:** Fine-tune selected models on the custom dataset to improve restoration quality on specific degradation types, especially under real-world conditions.
  - **Quantitative Evaluation:** Evaluate all models using standaardized face restoration metrics, including:
  **PSNR**,
  **SSIM**,
  **LPIPS**,
  **Deg**,
  **FID**,
  **identity Similarity(e.g, ArcFace-based cosine Similarity)**
  - **Model Selection:** Based on quantitative results and visual assessments, identity the most effective models for real-world blind face restoration.


## References

* **InstantRestore:**
  Snap Research, *InstantRestore: Single-Step Personalized Face Restoration*, 2024.
  [GitHub](https://github.com/snap-research/InstantRestore) | [Project Page](https://snap-research.github.io/InstantRestore/)

* **GFPGAN:**
  Wang et al., *GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior*, 2021.
  [arXiv](https://arxiv.org/abs/2101.04061) | [GitHub](https://github.com/TencentARC/GFPGAN)

* **CODEFORMER:**
  Zhou et al., *Towards Robust Blind Face Restoration with Codebook Lookup Transformer*, 2022.
  [arXiv](https://arxiv.org/abs/2206.11253) | [GitHub](https://github.com/sczhou/CodeFormer)

* **DFDNet:**
  Li et al., *Blind Face Restoration via Deep Multi-scale Component Dictionaries*, 2020.
  [arXiv](https://arxiv.org/abs/2008.00418) | [GitHub](https://github.com/csxmli2016/DFDNet)

* **DifFace:**
  Yue and Loy, *DifFace: Blind Face Restoration with Diffused Error Contraction*, 2022.
  [arXiv](https://arxiv.org/abs/2212.06512) | [GitHub](https://github.com/zsyOAOA/DifFace)

* **PMRF:**
  Saharia et al., *Posterior-Mean Rectified Flow for Image Restoration*, 2021.
  [arXiv](https://arxiv.org/abs/2112.03563) | [GitHub](https://github.com/ohayonguy/PMRF)
