# FastCap: Installation Guide

This guide provides detailed instructions for setting up the environment to run, train, and evaluate the FastCap model.

## 1. Prerequisites

* Python 3.8 or higher
* PyTorch 1.10 or higher (with CUDA support for GPU training)
* Git

## 2. Cloning the Repository

Clone the official repository from GitHub to your local machine:

```bash
git clone https://github.com/your-username/fastcap.git
cd fastcap
```

## 3. Setting up a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

### Using venv (standard Python)

```bash
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS and Linux:
source venv/bin/activate
```

### Using conda (recommended for PyTorch)

```bash
conda create -n fastcap python=3.9
conda activate fastcap
```

## 4. Installing Dependencies

This project has several dependencies, including PyTorch, Hugging Face datasets, and pycocotools.

### Step 4a: Install PyTorch

Install the appropriate version of PyTorch for your system (CPU or GPU). Visit the official PyTorch website to get the correct command for your setup.

Example for CUDA 11.8:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Example for CPU-only:

```bash
pip3 install torch torchvision torchaudio
```

### Step 4b: Install Project Requirements

Install all other required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Special Note on pycocotools Installation

The `pycocotools` library can sometimes fail to build during a standard `pip install`. If you encounter errors, you may need to install it with a pre-compiled binary or from a specific source.

For Windows:

```bash
pip install pycocotools-windows
```

For Linux (ensure C++ build tools are installed):

```bash
sudo apt-get install build-essential
pip install pycocotools
```

For macOS (ensure Xcode command-line tools are installed):

```bash
xcode-select --install
pip install pycocotools
```

If issues persist, consult the official installation instructions for `pycocotools`.

## 5. Verifying the Installation

Run a quick check by importing the necessary libraries in a Python shell:

```bash
python -c "import torch; import datasets; import transformers; from pycocoevalcap.cider.cider import Cider; print('All major libraries imported successfully!')"
```

If this command runs without errors, your environment is set up correctly.

## 6. Downloading the Dataset

Download the `yerevann/coco-karpathy` dataset from Hugging Face. The provided script will handle downloading and caching it in the default Hugging Face cache directory:

```bash
bash data/download_coco.sh
```

---

*End of installation guide.*
