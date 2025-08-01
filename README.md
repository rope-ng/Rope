# Rope-NG - Face Swapping Tool

A face swapping application with GUI interface, optimized for modern CUDA environments.

## About

This project is forked from [Hillobar/Rope](https://github.com/Hillobar/Rope) and has been specifically optimized for the newest CUDA versions (tested with CUDA 12.8). For original description, please visit the original repo.

## Features

- **Multiple Resolution Support**: Selectable model swapping output resolution (128, 256, 512)
- **Advanced Image Selection**: Enhanced input image selection with ctrl and shift modifiers
- **Real-time Preview**: Toggle between mean and median merging without saving
- **Keyboard Controls**: Full keyboard support (q, w, a, s, d, space)
- **Post-processing Options**: 
  - GFPGAN enhancement
  - Codeformer enhancement
  - CLIP-based processing
  - Occluder support
  - MouthParser integration
- **Performance Optimizations**: Multi-threaded processing with CUDA acceleration
- **Gamma Correction**: Adjustable gamma slider for fine-tuning

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.8 support (recommended)
- **Python**: 3.10 or higher
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **VRAM**: 12GB+ for optimal performance, should work on less.

## Installation

### 1. Create Virtual Environment

```bash
# Create a conda virtual environment with the correct Python version
conda create -n Rope python=3.13

# Activate the virtual environment
conda activate Rope

# Install the dependencies
python -m pip install -r requirements.txt
```

### 2. Download Models

Download all models from [https://github.com/rope-ng/Rope/releases/tag/Diamond](https://github.com/rope-ng/Rope/releases/tag/Diamond) and place them into the `models/` folder.

## Usage

Launch the application by running:
```bash
python Rope.py
```

The GUI provides intuitive controls for:
- Loading source and target media
- Configuring swap parameters
- Applying post-processing effects
- Rendering final output

## Disclaimer

This software is intended for responsible and ethical use only. Users are solely responsible for their actions when using this software.

### Intended Usage
This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. Users should explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

### Ethical Guidelines
Users are expected to adhere to ethical guidelines including:
- Not creating or sharing content that could harm, defame, or harass individuals
- Obtaining proper consent and permissions from individuals featured in the content
- Avoiding deceptive purposes, including misinformation or malicious intent
- Respecting and abiding by applicable laws, regulations, and copyright restrictions

### Privacy and Consent
Users are responsible for ensuring they have necessary permissions and consents from individuals whose likeness they intend to use. We strongly discourage creating content without explicit consent, particularly involving non-consensual or private content.

### Legal Considerations
Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology, including laws related to privacy, defamation, intellectual property rights, and other relevant legislation.

### Liability and Responsibility
The creators and providers of this software cannot be held responsible for actions or consequences resulting from software usage. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with their created content.

By using this software, users acknowledge they have read, understood, and agreed to abide by these guidelines and disclaimers. We encourage users to approach this technology with caution, integrity, and respect for others' well-being and rights.

Technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of face swapping technology.