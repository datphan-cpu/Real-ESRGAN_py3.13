<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

🚀 **Python 3.13 & PyTorch 2.x Standalone Edition** 🚀

👀[**Demos**](#-demos-videos) **|** ⚡[**Quick Start**](#-quick-inference) **|** 🏰[**Model Zoo**](docs/model_zoo.md) **|** 🔧[**Installation**](#-dependencies-and-installation) **|** 📓[**Colab**](#-google-colab-notebook) **|** 💻[**Training**](docs/Training.md) **|** 🎨[**Credits & Original Work**](#-credits--original-authors)

</div>

---

### 🌟 About This Fork (Python 3.13+ & PyTorch 2.x Compatible)

This repository is an updated, standalone distribution of **Real-ESRGAN** that solves compatibility breaks when running modern Python and PyTorch environments:
- 🐍 **Full Python 3.13+ Support**: Removed legacy C-extensions and obsolete Python APIs (e.g. `collections.Mapping`, older NumPy C API bindings).
- 🔥 **PyTorch 2.x+ & Safe Checkpoint Loading**: Supports latest PyTorch runtimes with fallbacks for modern serialization changes (`weights_only=False` where needed).
- 📦 **Decoupled from Deprecated external `basicsr`**: Core BasicSR components (such as `RRDBNet`, data degradations, USM sharpening, and registry loaders) are vendored locally in [`basicsr_local.py`](basicsr_local.py) and forwarded via local shims.
- 👤 **Local Face Restoration Pipeline**: Both **GFPGAN** and **facexlib** are integrated locally, allowing `--face_enhance` to operate standalone with weights stored in local directories without broken pip package installations.
- 📓 **Unified Colab Notebook**: An updated Colab notebook in `colab/` supporting dynamic switching between **Single Image** and **Full Video** upscaling.

---

### 📖 Original Research & Publications

> **Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data**  
> *Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan*  
> Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences  
> [[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [[YouTube Video](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[Bilibili](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[Project Slides](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

## 🔧 Dependencies and Installation

### 1. Requirements
- **Python**: >= 3.9 (Fully tested and verified on **Python 3.13**)
- **PyTorch**: >= 2.0 (Tested on **PyTorch 2.14+**)
- **torchvision**
- **CUDA GPU** (Optional for GPU acceleration, CPU inference also supported)

### 2. Setup

Clone the repository:
```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
```

Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

Install modern dependencies (no external `basicsr`, `gfpgan`, or `facexlib` pip builds required):
```bash
pip install -r requirements.txt
pip install torchvision
python setup.py develop
```

---

## ⚡ Quick Inference

Pre-trained weights are automatically downloaded on first run to the local `weights/` directory, or can be downloaded manually.

### 1. General Image Restoration

```bash
# Standard 4x upscale (RRDBNet)
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results

# With Face Enhancement (via vendored GFPGAN & facexlib)
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results --face_enhance
```

### 2. Anime Illustration Models

```bash
# Anime illustration model (RealESRGAN_x4plus_anime_6B)
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs -o results

# Fast anime video model (realesr-animevideov3)
python inference_realesrgan.py -n realesr-animevideov3 -i inputs -o results
```

### 3. Video Upscaling

```bash
python inference_realesrgan_video.py -n realesr-animevideov3 -i input_video.mp4 -o results
```

### 4. Common CLI Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-i`, `--input` | Input image path, video path, or directory | `inputs` |
| `-o`, `--output` | Destination output directory | `results` |
| `-n`, `--model_name` | Model name (`RealESRGAN_x4plus`, `RealESRGAN_x4plus_anime_6B`, `realesr-animevideov3`, `realesr-general-x4v3`, etc.) | `RealESRGAN_x4plus` |
| `-s`, `--outscale` | Final upsampling scale factor | `4` |
| `--face_enhance` | Enable local GFPGAN face restoration | `False` |
| `-t`, `--tile` | Tile size to prevent CUDA OOM (e.g. `200` or `400`). `0` for whole image | `0` |
| `--tile_pad` | Tile padding | `10` |
| `--pre_pad` | Border padding | `0` |
| `--fp32` | Force FP32 full precision instead of FP16 half precision | `False` |
| `--ext` | Image output format (`auto`, `jpg`, `png`) | `auto` |

---

## 📂 Project Structure

```text
Real-ESRGAN_py3.13/
├── basicsr/                     # Local BasicSR forwarder package
├── basicsr_local.py             # Vendored BasicSR core for Python 3.13 & PyTorch 2.x
├── colab/                       # Unified Colab Notebooks
│   └── Video_Upscaler_Colab_(Real_ESRGAN).ipynb
├── facexlib/                    # Vendored face detection and parsing modules
│   └── weights/                 # Local facexlib checkpoints
├── gfpgan/                      # Vendored GFPGAN face restoration
│   └── weights/                 # Local GFPGAN checkpoints
├── realesrgan/                  # Core Real-ESRGAN model architectures & pipelines
│   ├── archs/
│   ├── data/
│   ├── models/
│   └── utils.py
├── inputs/                      # Sample inputs
├── weights/                     # Pre-trained models (.pth)
├── inference_realesrgan.py      # Standalone image inference CLI
├── inference_realesrgan_video.py# Standalone video inference CLI
├── requirements.txt             # Cleaned dependency requirements
└── setup.py                     # Modernized installer
```

---

## 📓 Google Colab Notebook

A modernized Google Colab notebook is available in [`colab/Video_Upscaler_Colab_(Real_ESRGAN).ipynb`](colab/Video_Upscaler_Colab_(Real_ESRGAN).ipynb).  
It features an **Input Type** dropdown (`Image` vs `Video`), ensuring static images remain pristine `.png` files without requiring ffmpeg container wrapping.

---

## 🏰 Model Zoo

Checkpoints can be automatically fetched or placed into `weights/`:
- **RealESRGAN_x4plus**: [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- **RealESRNet_x4plus**: [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)
- **RealESRGAN_x4plus_anime_6B**: [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)
- **RealESRGAN_x2plus**: [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- **realesr-animevideov3**: [Download](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth)
- **GFPGANv1.3**: [Download](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

---

## 🎨 Credits & Original Authors

All original algorithms, model architectures, training configurations, and pre-trained weights were developed and published by the original authors:

- **Original Authors**: [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)
- **Organizations**: Tencent ARC Lab; Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
- **Original Repositories**:
  - [Real-ESRGAN (GitHub)](https://github.com/xinntao/Real-ESRGAN)
  - [BasicSR (GitHub)](https://github.com/xinntao/BasicSR)
  - [GFPGAN (GitHub)](https://github.com/TencentARC/GFPGAN)
  - [facexlib (GitHub)](https://github.com/xinntao/facexlib)

### BibTeX Citation
```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

### Community Acknowledgements
Thanks to the original contributors and community developers:
- [AK391](https://github.com/AK391): Gradio / HuggingFace demo integration.
- [Asiimoviet](https://github.com/Asiimoviet): Chinese documentation translation.
- [2ji3150](https://github.com/2ji3150): Feedback & suggestions.
- [Jared-02](https://github.com/Jared-02): Training guide translation.
- Authors and developers of GUI frontends: [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI), [Upscayl](https://github.com/upscayl/upscayl), [Real-GUI](https://github.com/scifx/Real-GUI), [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI).
