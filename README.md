# Face Restoration Comparison

A collection of Google Colab notebooks implementing five face restoration/deblurring models:
GFPGAN, PSFRGAN, DFDNet, DeblurGAN-v2, and PMRF. Each notebook allows you to test restoration quality on the same input images and compare results seamlessly.

---

## 1. Models & References

- **GFPGAN** — *Towards Real-World Blind Face Restoration with Generative Facial Prior* (leverages rich generative priors from a pretrained face GAN via channel-split spatial feature transform layers for detailed, color-enhanced single-pass restoration) :contentReference[oaicite:0]{index=0}.
- **PSFRGAN** — *Progressive Semantic-Aware Style Transformation for Blind Face Restoration* (utilizes multi-scale, semantic parsing map guidance for high-fidelity texture recovery).
- **DFDNet** — *Blind Face Restoration via Deep Multi-scale Component Dictionaries* (employs component-level dictionaries to reconstruct facial details progressively).
- **DeblurGAN-v2** — *Deblurring (Orders-of-Magnitude) Faster and Better* (introduces a relativistic conditional GAN with a dual-scale discriminator and Feature Pyramid Network for flexible backbones—achieving up to 10–100× faster deblurring while retaining near state-of-the-art quality) :contentReference[oaicite:1]{index=1}.
- **PMRF** — *(Your own model—please describe its methodology, motivation, and any internal or external references.)*

---

## 2. Comparison Table

| Model         | Strengths                                     | Limitations                          |
|---------------|-----------------------------------------------|--------------------------------------|
| GFPGAN        | Realistic, identity-preserving, color-rich     | May introduce hallucination artifacts |
| PSFRGAN       | Semantic-detail enhancement using parsing maps | Depends on accurate parsing maps     |
| DFDNet        | Interpretable, component-based restoration     | Limited to facial components, needs dictionary |
| DeblurGAN-v2  | Fast, flexible, general-purpose deblurring     | Not specialized for fine facial detail |
| PMRF          | *(Your observations—e.g. excels at…)*          | *(Limitations you’ve noticed)*       |

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

Use the following template to present visual comparisons:

## Sample Restoration Comparisons

| Original Image | GFPGAN Output | PSFRGAN Output | DFDNet Output | DeblurGAN-v2 Output | PMRF Output |
|----------------|----------------|----------------|----------------|----------------------|-------------|
| ![1](https://github.com/user-attachments/assets/3e87edb6-bf80-4938-84dd-236d347d710f)
   | ![1](https://github.com/user-attachments/assets/f69b4b46-b382-4877-81ed-4ae4219ee0b8)
`` | `![PSFR1](path/to/psfr1.png)` | `![DFD1](path/to/dfd1.png)` | `![Deblur1](path/to/deblur1.png)` | `![PMRF1](path/to/pmrf1.png)` |
| Sample 2       | `![GFPGAN2](...)` | ...            | ...            | ...                  | ...         |
| Sample 3       | ...            | ...            | ...            | ...                  | ...         |
| Sample 4       | ...            | ...            | ...            | ...                  | ...         |
| Sample 5       | ...            | ...            | ...            | ...                  | ...         |
