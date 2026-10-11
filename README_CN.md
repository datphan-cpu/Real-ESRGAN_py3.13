<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

<div align="center">

🚀 **Python 3.13 与 PyTorch 2.x 独立适配版** 🚀

👀[**演示**](#-演示视频) **|** ⚡[**快速开始**](#-快速推理) **|** 🏰[**模型库**](docs/model_zoo.md) **|** 🔧[**依赖与安装**](#-依赖与安装) **|** 📓[**Colab**](#-google-colab-笔记本) **|** 💻[**模型训练**](docs/Training.md) **|** 🎨[**致谢与原作者**](#-致谢与原作者)

</div>

---

### 🌟 本版本更新说明 (适配 Python 3.13+ 与 PyTorch 2.x)

本仓库是针对 **Real-ESRGAN** 在现代 Python 与 PyTorch 环境下构建问题的完全重构与本地化独立版本：
- 🐍 **原生支持 Python 3.13+**：修复了旧版本 C-API、NumPy 2.x 接口变更以及 `collections.Mapping` 移除导致的报错。
- 🔥 **原生支持 PyTorch 2.x+**：适配了新版本 PyTorch 的序列化权重加载机制（提供 `weights_only=False` 自动回退支持）。
- 📦 **完全解耦外部废弃的 `basicsr`**：将推理与数据处理所需的最小核心功能（如 `RRDBNet`、退化模拟、USM锐化）本地化提取至 [`basicsr_local.py`](basicsr_local.py)，摆脱了 broken 的 pip 外部包依赖。
- 👤 **本地人脸修复集成**：内置并解耦 **GFPGAN** 和 **facexlib**，支持在本地路径运行 `--face_enhance`，无需安装任何冲突的第三方旧包。
- 📓 **Colab 笔记本多模态重构**：提供了在 `colab/` 目录下的增强版 Notebook，支持图片与视频一键下拉切换，图片保留最高画质 `.png` 导出。

---

### 📖 原论文与研究成果

> **Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data**  
> *王新涛 (Xintao Wang), 谢良斌 (Liangbin Xie), 董超 (Chao Dong), 单瀛 (Ying Shan)*  
> 腾讯 ARC 实验室; 中国科学院深圳先进技术研究院  
> [[论文](https://arxiv.org/abs/2107.10833)] &emsp; [[YouTube 视频](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[B站视频](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[PPT 演示](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

## 🔧 依赖与安装

### 1. 环境要求
- **Python**: >= 3.9 (在 **Python 3.13** 上全面测试通过)
- **PyTorch**: >= 2.0 (在 **PyTorch 2.14+** 上全面测试通过)
- **torchvision**
- **Nvidia GPU** (推荐用于硬件加速，亦可使用 CPU 推理)

### 2. 本地安装步骤

克隆本仓库：
```bash
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
```

创建并激活虚拟环境：
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

安装纯净依赖（无需编译外部 `basicsr` 或安装旧版 `gfpgan`/`facexlib`）：
```bash
pip install -r requirements.txt
pip install torchvision
python setup.py develop
```

---

## ⚡ 快速推理

预训练权重在首次调用时会自动下载至本地 `weights/` 目录，也可手动提前放置。

### 1. 通用图像超分辨率

```bash
# 标准 4倍 超分辨率 (RRDBNet 模型)
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results

# 启用人脸修复 (使用内置的本地 GFPGAN + facexlib)
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs -o results --face_enhance
```

### 2. 动漫图像与视频模型

```bash
# 动漫插画小模型 (RealESRGAN_x4plus_anime_6B)
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs -o results

# 高速动漫视频模型 (realesr-animevideov3)
python inference_realesrgan.py -n realesr-animevideov3 -i inputs -o results
```

### 3. 视频修复推理

```bash
python inference_realesrgan_video.py -n realesr-animevideov3 -i input_video.mp4 -o results
```

### 4. 常用命令行参数说明

| 参数 | 详细说明 | 默认值 |
| :--- | :--- | :--- |
| `-i`, `--input` | 输入图片、视频文件或目录 | `inputs` |
| `-o`, `--output` | 输出文件夹 | `results` |
| `-n`, `--model_name` | 模型名称 (`RealESRGAN_x4plus`, `RealESRGAN_x4plus_anime_6B`, `realesr-animevideov3` 等) | `RealESRGAN_x4plus` |
| `-s`, `--outscale` | 最终输出图像的缩放比例 | `4` |
| `--face_enhance` | 开启本地 GFPGAN 进行人脸增强 | `False` |
| `-t`, `--tile` | 切块大小以防止显存溢出 (如 `200` 或 `400`)，`0` 表示整图推理 | `0` |
| `--fp32` | 强制使用 FP32 全精度推理（默认使用半精度 FP16） | `False` |
| `--ext` | 输出格式 (`auto`, `jpg`, `png`) | `auto` |

---

## 📂 项目结构说明

```text
Real-ESRGAN_py3.13/
├── basicsr/                     # 本地 BasicSR 接口映射模块
├── basicsr_local.py             # 针对 Python 3.13 优化的本地 BasicSR 核心
├── colab/                       # 增强版 Google Colab 笔记本
│   └── Video_Upscaler_Colab_(Real_ESRGAN).ipynb
├── facexlib/                    # 本地集成的人脸检测与分割库
│   └── weights/                 # 本地 facexlib 模型权重存储
├── gfpgan/                      # 本地集成的 GFPGAN 人脸增强库
│   └── weights/                 # 本地 GFPGAN 模型权重存储
├── realesrgan/                  # Real-ESRGAN 核心网络架构与算法
├── inputs/                      # 示例输入文件目录
├── weights/                     # 预训练权重存放目录
├── inference_realesrgan.py      # 独立图像超分脚本
├── inference_realesrgan_video.py# 独立视频超分脚本
├── requirements.txt             # 优化后的依赖列表
└── setup.py                     # 独立安装配置脚本
```

---

## 📓 Google Colab 笔记本

在 [`colab/Video_Upscaler_Colab_(Real_ESRGAN).ipynb`](colab/Video_Upscaler_Colab_(Real_ESRGAN).ipynb) 中提供了最新可运行的 Google Colab 脚本。该脚本支持通过下拉菜单自由选择处理 **图片 (Image)** 还是 **视频 (Video)**，图片模式下将自动导出无损 `.png` 格式。

---

## 🎨 致谢与原作者

本项目所有原始算法、神经网络设计、训练策略和预训练权重均归原作者所有：

- **原作者**: [王新涛 (Xintao Wang)](https://xinntao.github.io/), 谢良斌 (Liangbin Xie), [董超 (Chao Dong)](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [单瀛 (Ying Shan)](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)
- **研发机构**: 腾讯 ARC 实验室; 中国科学院深圳先进技术研究院
- **官方开源项目**:
  - [Real-ESRGAN (GitHub)](https://github.com/xinntao/Real-ESRGAN)
  - [BasicSR (GitHub)](https://github.com/xinntao/BasicSR)
  - [GFPGAN (GitHub)](https://github.com/TencentARC/GFPGAN)
  - [facexlib (GitHub)](https://github.com/xinntao/facexlib)

### BibTeX 引用
```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    date      = {2021}
}
```

### 社区致谢
感谢开源社区各位贡献者的支持与贡献：
- [AK391](https://github.com/AK391): 搭建 Gradio / HuggingFace Web Demo。
- [Asiimoviet](https://github.com/Asiimoviet): 中文文档初版翻译。
- [2ji3150](https://github.com/2ji3150): 宝贵的问题反馈与建议。
- [Jared-02](https://github.com/Jared-02): 训练教程中文翻译。
- 各大前端工具开发者: [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI), [Upscayl](https://github.com/upscayl/upscayl), [Real-GUI](https://github.com/scifx/Real-GUI), [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI).
