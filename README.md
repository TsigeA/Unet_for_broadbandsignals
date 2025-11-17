# UNet for Broadband Signal Segmentation

Semantic segmentation of broadband signals in spectrograms using a U-Net deep learning model. The model identifies and segments broadband signal regions in radio frequency spectrograms.

## Overview

- **Model**: U-Net (encoder-decoder with skip connections)
- **Task**: Binary semantic segmentation of spectrograms
- **Input**: Grayscale spectrograms (1 channel, H×W)
- **Output**: Binary segmentation masks (0 = background, 1 = broadband signal)
- **Loss**: BCEWithLogitsLoss (numerically stable binary cross-entropy)

## Dataset

- Format: `.npz` files with keys `"spec"` and `"mask"`
- Spectrogram shape: (H, W) or (H, W, 1) or (H, W, 3)
- Mask shape: (H, W) or (H, W, 1) with binary labels {0, 1}
- Preprocessing: optional z-score normalization, min-max normalization, or resizing

## Setup

### Requirements
- Python 3.9+
- PyTorch with CUDA support
- torchvision, torchaudio
- torchmetrics
- numpy, cv2

### Installation

```bash
# Create conda environment (optional)
conda create -n unet-broadband python=3.11
conda activate unet-broadband

# Install dependencies
pip install torch torchvision torchaudio torchmetrics opencv-python numpy
```

### Project Structure

```
Unet_for_broadbandsignals/
├── unet_obsdata_2.py          # Main training script
├── unet/                        # U-Net model implementation
│   ├── __init__.py
│   └── unet_model.py
├── masks_11052025/             # Dataset folder (.npz files)
├── prediction_plots/           # Output predictions and logs
├── README.md
└── .gitignore
```

## Usage

### 1. Prepare Dataset

Place `.npz` files in the `masks_11052025/` folder (or update `data_path` in the script).

Each `.npz` file should contain:
```python
{
    "spec": numpy array (H, W) or (H, W, 1) or (H, W, 3),
    "mask": numpy array (H, W) or (H, W, 1) with binary labels
}
```

### 2. Train the Model

```bash
python unet_obsdata_2.py
```

**Configuration** (edit in `__main__`):
- `data_path`: folder containing `.npz` files
- `n_channel`: input channels (1 for grayscale)
- `n_class`: output classes (1 for binary segmentation)
- `batch_size`: training batch size
- `epochs`: number of training epochs
- `learningrate`: optimizer learning rate

### 3. Training Output

- **Models**: saved to `prediction_plots/run_obsdata<YYYYMMDD_HHMM>/unet_model_obsdata*.pth`
- **Plots**: predicted masks saved as `.png` in `prediction_plots/`
- **Logs**: training hyperparameters and metrics in `.txt` file

### 4. Results

Metrics computed on test set (20% of data):
- **Dice Coefficient**: F1-score equivalent for segmentation
- **IoU (Intersection over Union)**: Jaccard index
- **Accuracy**: per-pixel classification accuracy
- **Recall**: sensitivity (true positive rate)
- **Precision**: positive predictive value
- **F1 Score**: harmonic mean of precision and recall

## Model Architecture

U-Net encoder-decoder:
- **Encoder**: 4 downsampling blocks (Conv → BatchNorm → ReLU → MaxPool)
- **Bottleneck**: 1 downsampling block (deepest feature maps)
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 1×1 convolution → logits (binary segmentation)

**Skip Connections**: preserve spatial information from encoder layers during upsampling.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_channel` | 1 | Input channels (1 = grayscale) |
| `n_class` | 1 | Output channels (1 = binary segmentation) |
| `batch_size` | 4 | Training batch size |
| `epochs` | 68 | Number of training epochs |
| `learningrate` | 1e-3 | Adam optimizer learning rate |
| `target_shape` | None | Resize spectrograms to (H, W); None keeps native size |
| `normalization` | None | Preprocessing: "zscore", "minmax", or None |
| `binary_threshold` | 0.5 | Threshold for binarizing predictions (0–1) |

## Dataset Class

`SpectrogramSegmentationDataset`:
- Loads `.npz` files from a folder
- Converts spectrograms to (C, H, W) tensors
- Matches channel count (1 or 3) and normalizes
- Optionally resizes using bilinear (images) and nearest-neighbor (masks)

## Training Details

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: BCEWithLogitsLoss (combines sigmoid + BCE for numerical stability)
- **Validation**: 80/20 train/test split
- **Device**: GPU (CUDA) if available, else CPU

## Troubleshooting

### Import Errors
- Ensure U-Net module is in `unet/` with `__init__.py`
- Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`

### Dataset Not Found
- Verify `.npz` files exist in `data_path`
- Print dataset size: `print(f"Dataset size: {len(dataset)}")`

## References
# model addapted from
- PyTorch UNet: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)


## Author

Tsige Atilaw  
Date: 2025-08-11
