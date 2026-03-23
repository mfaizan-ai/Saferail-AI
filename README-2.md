# 🚆 Saferail-AI

<p align="center">
  <img src="https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20Orin%20Nano-green?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-orange?style=for-the-badge" />
  <img src="https://img.shields.io/github/stars/mfaizan-ai/Saferail-AI?style=for-the-badge" />
</p>

> **AI-powered railway level crossing safety system** using infrared-optical image fusion, real-time object detection, depth estimation, and ROI segmentation — built for Pakistan Railway.

---

## 📖 Table of Contents

- [Background](#-background)
- [How It Works](#-how-it-works)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Model Weights](#-model-weights)
- [Deployment on Jetson](#-deployment-on-nvidia-jetson-orin-nano)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚨 Background

Pakistan Railway is the country's most affordable means of public transport, carrying over **52.2 million passengers** annually and averaging **178,000 passengers per day** across 28 mail, express, and passenger train lines.

Despite its critical role, safety remains a serious concern. Over a five-year period, **537 train accidents** occurred — 313 of which resulted in loss of life or serious injury. Factorial analysis reveals that **32% of these accidents** happened at **unmanned level crossings**, where the responsibility fell on road users.

Traditional optical camera-based detection systems fail under adverse conditions — **low light, fog, rain, and nighttime** — leaving a dangerous blind spot in the safety net.

**Saferail-AI** was built to close that gap.

---

## ⚙️ How It Works

Saferail-AI combines **infrared (thermal)** and **optical (RGB)** camera feeds to overcome the limitations of single-modality vision:

1. **Infrared Imaging** — Captures ambient thermal radiation emitted by objects, making them visible regardless of lighting conditions.
2. **Image Fusion (TarDAL)** — Fuses infrared and optical frames to produce rich, all-weather imagery that preserves both thermal and textural detail.
3. **Object Detection (YOLOv5)** — Runs real-time detection on the fused frames to identify people, vehicles, and obstacles on or near the track.
4. **Depth / Distance Estimation** — Estimates the distance of detected objects from the train's front camera.
5. **ROI Segmentation** — Isolates the track region of interest to filter irrelevant detections and focus on objects that pose an actual collision risk.
6. **MMI Driver Alert** — Displays classified object type and its distance on a Man–Machine Interface (MMI) screen in the driver's cabin, triggering timely warnings and control actions.

---

## ✨ Key Features

- 🌑 **All-weather, day/night operation** via infrared-optical sensor fusion
- 🎯 **Real-time object detection** optimised for edge hardware
- 📏 **Distance estimation** to quantify collision risk
- 🛤️ **Track ROI segmentation** to eliminate false positives
- ⚡ **TensorRT / INT8 quantisation** support for Jetson deployment
- 📡 **RTSP stream support** for live camera feeds
- 🔌 **Socket-based streaming** for remote MMI display
- 📊 **ONNX export** for cross-platform model portability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Saferail-AI Pipeline                    │
│                                                             │
│  [RGB Camera] ──┐                                           │
│                 ├──► [Image Fusion (TarDAL)] ──►            │
│  [IR Camera]  ──┘         ↓                                 │
│                    [Fused Frame]                             │
│                         │                                   │
│                         ▼                                   │
│              [YOLOv5 Object Detection]                       │
│                         │                                   │
│              ┌──────────┴──────────┐                        │
│              ▼                     ▼                        │
│    [ROI Segmentation]    [Depth Estimation]                  │
│              │                     │                        │
│              └──────────┬──────────┘                        │
│                         ▼                                   │
│              [MMI Display + Driver Alert]                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Edge SoC** | NVIDIA Jetson Orin Nano 8GB |
| **RGB Camera** | Any compatible USB / CSI camera |
| **Infrared Camera** | Thermal/IR camera with video output |
| **Storage** | ≥ 32 GB (microSD or NVMe SSD recommended) |
| **OS** | Ubuntu 20.04 / JetPack 5.x |

> **Note:** The system can also be run on a standard desktop/laptop GPU for development and testing. Jetson-specific instructions are provided in the [Deployment section](#-deployment-on-nvidia-jetson-orin-nano).

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- `git` with submodule support
- CUDA toolkit compatible with your environment

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/mfaizan-ai/Saferail-AI
cd Saferail-AI
```

> If you already cloned without `--recurse-submodules`, initialise submodules manually:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Create a Virtual Environment

```bash
python3 -m venv saferail
source saferail/bin/activate        # Linux / macOS
# saferail\Scripts\activate.bat     # Windows
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install PyTorch

PyTorch must be installed separately to match your hardware:

**Standard GPU (desktop/laptop):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**NVIDIA Jetson (JetPack):**
Follow the official NVIDIA guide for your JetPack version:
👉 https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html

---

## 🚀 Usage

### Run the Main Application

```bash
python app_final.py
```

### Run with RTSP Camera Stream

```bash
python app_final_rtsp.py --source rtsp://<camera-ip>:<port>/stream
```

### Run with Socket Streaming (for Remote MMI)

```bash
python app_socket_final.py --host <mmi-host-ip> --port 5005
```

### Track Objects in a Region of Interest

```bash
python track_roi.py --source <video_path_or_camera_index>
```

### Export Model to ONNX

```bash
python generate_onnx.py --weights detection_weights/weights/<model>.pt --output model.onnx
```

### Run TensorRT Inference

```bash
python run_trt_inference.py --engine model.engine --source <video_path>
```

---

## 📁 Project Structure

```
Saferail-AI/
│
├── app.py                     # Basic application entry point
├── app_final.py               # Main production application
├── app_final_rtsp.py          # RTSP stream variant
├── app_socket_final.py        # Socket-based streaming variant
├── track_roi.py               # ROI-based object tracking
├── generate_onnx.py           # ONNX model export utility
├── run_trt_inference.py       # TensorRT inference runner
├── utils.py                   # Shared utility functions
├── requirements.txt           # Python dependencies
│
├── configs/                   # YAML configuration files
├── fusion/                    # Image fusion module
├── TarDAL/                    # Submodule: IR-optical fusion network
├── detection_segmentation/    # Object detection & segmentation scripts
├── detection_weights/
│   └── weights/               # YOLOv5 detection model weights
├── segmentation_weights/
│   └── pt_weights/            # Segmentation model weights
├── int8_calibration/          # INT8 quantisation calibration data
├── registration_data/         # Camera registration / calibration data
├── JetsonYolov5               # Jetson-optimised YOLOv5 deployment
├── scripts/                   # Helper and setup scripts
└── videos/                    # Sample test videos
```

---

## ⚙️ Configuration

All configurable parameters live in the `configs/` directory as YAML files. Key settings include:

| Parameter | Description |
|-----------|-------------|
| `detection.conf_threshold` | Minimum confidence score for object detection |
| `detection.iou_threshold` | IoU threshold for NMS |
| `fusion.weights` | Path to TarDAL fusion model weights |
| `camera.rgb_source` | RGB camera index or RTSP URL |
| `camera.ir_source` | Infrared camera index or RTSP URL |
| `depth.model` | Depth estimation model selection |
| `roi.polygon` | Track ROI polygon coordinates |

Edit the relevant config file before running:

```bash
nano configs/saferail_config.yaml
```

---

## 📦 Model Weights

Place pre-trained weights in the appropriate directories before running:

| Model | Directory |
|-------|-----------|
| YOLOv5 detection weights | `detection_weights/weights/` |
| Segmentation weights | `segmentation_weights/pt_weights/` |
| TarDAL fusion weights | `TarDAL/` (managed as submodule) |

Refer to each module's documentation or contact the maintainers for access to trained weights.

---

## 🔧 Deployment on NVIDIA Jetson Orin Nano

### 1. Flash JetPack

Download and flash JetPack 5.x from the [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager).

### 2. Install Jetson-specific PyTorch

```bash
# Follow NVIDIA's official guide:
# https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
```

### 3. Convert Model to TensorRT Engine

```bash
python generate_onnx.py --weights detection_weights/weights/best.pt
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

For INT8 quantisation (maximum performance):

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --int8 \
        --calib=int8_calibration/
```

### 4. Run TensorRT Inference on Jetson

```bash
python run_trt_inference.py --engine model.engine --source /dev/video0
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open a Pull Request**

Please make sure your code follows the existing style and includes appropriate comments.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL) — Infrared and visible image fusion network
- [YOLOv5](https://github.com/ultralytics/yolov5) — Real-time object detection
- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-orin-nano-devkit) — Edge AI hardware platform
- Pakistan Railway — For inspiring and motivating this safety-critical work

---

<p align="center">
  Made with ❤️ to make Pakistan's railways safer.
</p>
