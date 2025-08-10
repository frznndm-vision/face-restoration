This repository provides a comprehensive suite of state-of-the-art face restoration and enhancement models for repairing and improving low-quality facial images. It is designed for researchers, developers, and enthusiasts who want to compare and apply multiple restoration methods in a single, easy-to-use framework.
✨ Features

    Multiple restoration approaches — From GAN-based generative priors to dictionary learning and semantic parsing.
    
    Modular architecture — Easily switch between models for experimentation or benchmarking.
    
    CPU and GPU support — Optimized configurations for different hardware setups.
    
    Before/After visual comparisons — Quickly evaluate restoration quality.


| Model           | Description                                                                                                                                         | Paper / Repo                                                                                                                                                                   | Highlights                                                                                                            |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **GFPGAN**      | Blind face restoration using a **Generative Facial Prior** to reconstruct realistic facial details while preserving identity.                       | [Repo](https://github.com/TencentARC/GFPGAN) · [Paper](https://arxiv.org/abs/2101.04061) — *Towards Real-World Blind Face Restoration with Generative Facial Prior*            | Excellent perceptual quality, strong real-world performance, robust to various degradations.                          |
| **PSFRGAN**     | Utilizes **face parsing maps** to guide GAN-based restoration, recovering fine-grained structures and textures.                                     | [Repo](https://github.com/chaofengc/PSFRGAN) · [Paper](https://arxiv.org/pdf/2009.08709) — *Face Parsing Assisted Blind Face Restoration in the Wild* (CVPR 2021)              | Strong semantic consistency, detail enhancement, and resilience to pose/expression changes.                           |
| **DFDNet**      | Leverages a **dictionary of high-quality facial components** to restore structure and preserve facial identity from degraded inputs.                | [Repo](https://github.com/csxmli2016/DFDNet) · [Paper](https://arxiv.org/pdf/2009.08709) — *Blind Face Restoration via Deep Multi-scale Component Dictionaries* (ECCV 2020)    | Effective for severely degraded images, high identity preservation.                                                   |
| **DeblurGANv2** | High-speed image deblurring using a **feature pyramid generator** and flexible backbones (MobileNet, Inception-ResNet).                             | [Repo](https://github.com/VITA-Group/DeblurGANv2) · [Paper](https://arxiv.org/abs/1908.03826) — *DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better*             | Extremely fast inference, adaptable to computational budgets, effective for motion and defocus blur.                  |
| **PMRF**        | Two-stage method combining **Posterior Mean estimation** (minimizing distortion) and **Rectified Flow refinement** (maximizing perceptual realism). | [Repo](https://github.com/ohayonguy/PMRF) · [Paper](https://arxiv.org/abs/2410.00418) — *Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration* | Theoretically grounded, achieves optimal distortion–perception trade-off, state-of-the-art in blind face restoration. |








| Input                       | GFPGAN                         | PSFRGAN                          | DFDNet                         | DeblurGANv2                                 | PMRF                       |
| --------------------------- | ------------------------------ | -------------------------------- | ------------------------------ | ------------------------------------------- | -------------------------- |
| ![input]() | | ![psfrgan1](images/psfrgan1.jpg) | ![dfdnet1](images/dfdnet1.jpg) | ![deblurganv2\_1](images/deblurganv2_1.jpg) | ![pmrf1](images/pmrf1.jpg) |
| ![input](images/input2.jpg) | ![gfpgan2](images/gfpgan2.jpg) | ![psfrgan2](images/psfrgan2.jpg) | ![dfdnet2](images/dfdnet2.jpg) | ![deblurganv2\_2](images/deblurganv2_2.jpg) | ![pmrf2](images/pmrf2.jpg) |
