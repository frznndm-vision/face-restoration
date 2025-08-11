# Face Restoration Comparison

A collection of Google Colab notebooks implementing five face restoration/deblurring models:
GFPGAN, PSFRGAN, DFDNet, DeblurGAN-v2, and PMRF. Each notebook allows you to test restoration quality on the same input images and compare results seamlessly.

---

## 1. Models & References

- **GFPGAN** — *Towards Real-World Blind Face Restoration with Generative Facial Prior* (leverages rich generative facial priors from pretrained GANs using channel-split spatial feature transform layers for detailed, color-enhanced single-pass restoration) :contentReference[oaicite:0]{index=0}.
- **PSFRGAN** — *Progressive Semantic-Aware Style Transformation for Blind Face Restoration* (utilizes multi-scale, semantic parsing map guidance for high-fidelity texture recovery).
- **DFDNet** — *Blind Face Restoration via Deep Multi-scale Component Dictionaries* (employs component-level dictionaries to reconstruct facial details progressively).
- **DeblurGAN-v2** — *Deblurring (Orders-of-Magnitude) Faster and Better* (introduces a relativistic conditional GAN with a double-scale discriminator and Feature Pyramid Network, achieving state-of-the-art performance and up to 100× faster inference with lightweight backbones) :contentReference[oaicite:1]{index=1}.
- **PMRF** — *(Your model—describe it concisely here, e.g., methodology, motivations, and whether there is an associated paper or internal reference.)*

---

## 2. Comparison Table

| Model         | Strengths                                  | Limitations                     |
|---------------|---------------------------------------------|----------------------------------|
| GFPGAN        | Realistic, identity-preserving, color-restoring | Potential hallucination artifacts |
| PSFRGAN       | Semantic-detail enhancement via parsing maps | Requires precise parsing inputs  |
| DFDNet        | Interpretable, component-based restoration   | Limited to face region, needs dictionaries |
| DeblurGAN-v2  | Fast, flexible, general-purpose              | Not tailored for fine facial detail |
| PMRF          | *(Your observations—e.g., stronger on X)*     | *(Limitations you observed)*     |

---

## 3. Usage Instructions (Google Colab)

Each model has its own Colab notebook (`GFPGAN_colab.ipynb`, `PSFRGAN_colab.ipynb`, etc.) containing:

1. **Setup**: Installs dependencies and clones the relevant repo.
2. **Model Loading**: Downloads and initializes pretrained weights.
3. **Inference**: Runs restoration on sample images or user uploads.
4. **Outputs**: Saves results under `outputs/<model>/`.

**To use:**

1. Open the desired `.ipynb` in Colab.
2. Select GPU as runtime type.
3. Upload input images into the designated input folder in Colab.
4. Run all cells to generate outputs. Outputs will appear in `outputs/<model>/`.

---

## 4. Sample Results Table

Compare the visual results across models using this table template:

```markdown
## Sample Restoration Comparisons

| Original Image | GFPGAN Output | PSFRGAN Output | DFDNet Output | DeblurGAN-v2 Output | PMRF Output |
|----------------|----------------|----------------|----------------|----------------------|-------------|
| Sample 1       | `![GFPGAN1](path/to/gfpgan1.png)` | `![PSFR1](path/to/psfr1.png)` | `![DFD1](path/to/dfd1.png)` | `![Deblur1](path/to/deblur1.png)` | `![PMRF1](path/to/pmrf1.png)` |
| Sample 2       | `![GFPGAN2](...)` | ...            | ...            | ...                  | ...         |
| Sample 3       | ...            | ...            | ...            | ...                  | ...         |
| Sample 4       | ...            | ...            | ...            | ...                  | ...         |
| Sample 5       | ...            | ...            | ...            | ...                  | ...         |
