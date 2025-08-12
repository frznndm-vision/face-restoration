# Face Restoration Comparison

A collection of Google Colab notebooks implementing five face restoration/deblurring models:
GFPGAN, PSFRGAN, DFDNet, DeblurGAN-v2, and PMRF. Each notebook allows you to test restoration quality on the same input images and compare results seamlessly.

---

## 1. Models & References

- **GFPGAN** — GFPGAN is a blind face restoration method that uses a pretrained face GAN as a Generative Facial Prior to restore realistic and faithful facial details, even from very low-quality inputs. By introducing novel channel-split spatial feature transform layers, it balances realness and fidelity in a single forward pass—no expensive per-image optimization required. GFP-GAN can restore fine details and enhance colors simultaneously, outperforming prior methods on both synthetic and real-world datasets.
- **PSFRGAN** — *Progressive Semantic-Aware Style Transformation for Blind Face Restoration* (utilizes multi-scale, semantic parsing map guidance for high-fidelity texture recovery).
- **DFDNet** — *Blind Face Restoration via Deep Multi-scale Component Dictionaries* (employs component-level dictionaries to reconstruct facial details progressively).
- **DeblurGAN-v2** — *Deblurring (Orders-of-Magnitude) Faster and Better* (introduces a relativistic conditional GAN with a dual-scale discriminator and Feature Pyramid Network for flexible backbones—achieving up to 10–100× faster deblurring while retaining near state-of-the-art quality) :contentReference[oaicite:1]{index=1}.
- **PMRF** — *(Your own model—please describe its methodology, motivation, and any internal or external references.)*

---

## 2. Comparison Table

| Model         | Strengths                                     | Limitations                          |
|---------------|-----------------------------------------------|--------------------------------------|
| GFPGAN        | Realistic, identity-preserving, color-rich results; efficient single-pass restoration; superior performance over earlier methods.     | Struggles with extremely degraded inputs and large pose variations; may introduce color bias when input lacks color information. |
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
|----------------|--------------|----------------|---------------|---------------------|-------------|
| ![1](https://github.com/user-attachments/assets/3e87edb6-bf80-4938-84dd-236d347d710f) | ![GFPGAN1](https://github.com/user-attachments/assets/f69b4b46-b382-4877-81ed-4ae4219ee0b8) | ![PSFR1](path/to/psfr1.png) | ![DFD1](path/to/dfd1.png) | ![Deblur1](path/to/deblur1.png) | ![PMRF1](path/to/pmrf1.png) |
| ![2](https://github.com/user-attachments/assets/cbae896d-fa8c-4e26-a8f6-c76b2dbd5567) |![2](https://github.com/user-attachments/assets/060fd1de-9829-4fdb-886a-59f56415ea96)
  | ![PSFR2](path/to/psfr2.png) | ![DFD2](path/to/dfd2.png) | ![Deblur2](path/to/deblur2.png) | ![PMRF2](path/to/pmrf2.png) |
| <img width="114" height="142" alt="3" src="https://github.com/user-attachments/assets/6670c5d6-af50-4534-9378-b3a7f97243f6" /> | ![GFPGAN3](path/to/gfpgan3.png) | ![PSFR3](path/to/psfr3.png) | ![DFD3](path/to/dfd3.png) | ![Deblur3](path/to/deblur3.png) | ![PMRF3](path/to/pmrf3.png) |
| ![4](https://github.com/user-attachments/assets/498c36b8-9f2c-4340-9285-625b44cbd234) | ![GFPGAN4](path/to/gfpgan4.png) | ![PSFR4](path/to/psfr4.png) | ![DFD4](path/to/dfd4.png) | ![Deblur4](path/to/deblur4.png) | ![PMRF4](path/to/pmrf4.png) |
| <img width="114" height="142" alt="7" src="https://github.com/user-attachments/assets/9333c797-2517-440b-8deb-2af083d583ef" /> | ![GFPGAN7](path/to/gfpgan7.png) | ![PSFR7](path/to/psfr7.png) | ![DFD7](path/to/dfd7.png) | ![Deblur7](path/to/deblur7.png) | ![PMRF7](path/to/pmrf7.png) |

