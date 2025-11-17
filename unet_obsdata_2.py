"""Using a Unet model to segment images.
- updated to calculate the metrics per each spectrogram then calculate the average
- using the Unet model from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
- adapt the dice calculation from https://github.com/milesial/Pytorch-UNet/blob/master/evaluate.py
@author: Tsige Atilaw
@date: 2025-08-11
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
# from torchinfo import summary
import torch.optim as optim
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
import glob
from unet import UNet
import logging
import random
import cv2  # For image processing
# from scipy.ndimage import zoom  # For resizing
import logging
from typing import Union, Tuple
from datetime import datetime
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryJaccardIndex,
)

# class SpectrogramSegmentationDataset(Dataset):
#     def __init__(self, folder, target_shape=(1024, 1000)):
#         self.files = sorted(glob.glob(f"{folder}/*.npz"))
#         self.target_shape = target_shape  # Target shape for resizing

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         data = np.load(self.files[idx])
#         # spec = (data["spec"] - np.mean(data["spec"])) / np.std(data["spec"])  # z-normalize
#         # make sure the spectrogram is in two dimensions (x,y)
#         if len(data["spec"].shape) == 3:  # If the spectrogram has a channel dimension
#             spec = data["spec"][:,:, 0]  # Take the first channel (assuming it's a single-channel spectrogram)  
#         else:
#             spec = data["spec"]  # Use the original spectrogram without normalization
        
#         mask = data["mask"]
#         # use resizing only for dataset in mask_3 folder
#         # datasets in mask_4 folder have spectrograms of same size     
#         # Resize spectrogram and mask to the target shape or the same size
#         # spec = cv2.resize(spec, (1024, 1000), interpolation=cv2.INTER_CUBIC)
#         # mask = cv2.resize(mask, (1024, 1000), interpolation=cv2.INTER_CUBIC)
#         # spec = self.resize_array(spec, self.target_shape)
#         # mask = self.resize_array(mask, self.target_shape)

#         # Convert to PyTorch tensors
#         spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#         mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

#         return spec, mask

#     # def resize_array(self, array, target_shape):
#     #     # Resize using numpy's slicing and padding
#     #     resized = np.zeros(target_shape, dtype=array.dtype)
#     #     min_shape = (
#     #         min(target_shape[0], array.shape[0]),
#     #         min(target_shape[1], array.shape[1]),
#     #     )
#     #     resized[:min_shape[0], :min_shape[1]] = array[:min_shape[0], :min_shape[1]]
#     #     return resized
###################################
import glob, numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpectrogramSegmentationDataset(Dataset):
    """
    Loads .npz files with keys: 'spec' and 'mask'.

    - spec: (H,W), (H,W,1) or (H,W,3) -> returns (C,H,W) float32
    - mask: (H,W) or (H,W,1) -> returns (1,H,W) float32 in {0,1}

    Args:
        folder: directory containing *.npz
        target_shape: optional (H, W). If provided, image is bilinear-resized,
                      mask is nearest-resized to this size.
        out_channels: 1 or 3. If 1 and input has 3 channels, convert to grayscale (mean).
                      If 3 and input has 1 channel, repeat to 3.
        normalization: "zscore", "minmax", or None (per-channel, per-sample).
    """
    def __init__(self, folder: str,
                 target_shape: Union[Tuple[int, int], None] = None,
                 out_channels: int = 1,
                 normalization: Union[str, None] = "zscore"): 
        self.files = sorted(glob.glob(f"{folder}/*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {folder}")
        assert out_channels in (1, 3), "out_channels must be 1 or 3"
        assert normalization in ("zscore", "minmax", None)
        self.target_shape = target_shape
        self.out_channels = out_channels
        self.normalization = normalization

    def __len__(self):
        return len(self.files)

    def _to_chw(self, arr: np.ndarray) -> torch.Tensor:
        """(H,W), (H,W,1) or (H,W,3) -> (C,H,W) float32"""
        if arr.ndim == 2:
            t = torch.from_numpy(arr).float().unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3):
            t = torch.from_numpy(arr).permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unsupported spec shape {arr.shape}")
        return t

    def _match_channels(self, t: torch.Tensor) -> torch.Tensor:
        """Make t have self.out_channels."""
        C, H, W = t.shape
        if self.out_channels == 1:
            if C == 1:
                return t
            elif C == 3:
                return t.mean(dim=0, keepdim=True)  # RGB -> grayscale
        else:  # out_channels == 3
            if C == 3:
                return t
            elif C == 1:
                return t.repeat(3, 1, 1)            # gray -> pseudo-RGB
        raise ValueError(f"Unexpected channel count: {C}")

    def _normalize(self, t: torch.Tensor) -> torch.Tensor:
        if self.normalization is None:
            return t
        # per-channel, per-sample
        if self.normalization == "zscore":
            mean = t.mean(dim=(1, 2), keepdim=True)
            std = t.std(dim=(1, 2), keepdim=True)
            t = (t - mean) / (std + 1e-8)
        elif self.normalization == "minmax":
            tmin = t.amin(dim=(1, 2), keepdim=True)
            tmax = t.amax(dim=(1, 2), keepdim=True)
            t = (t - tmin) / (tmax - tmin + 1e-8)
        return t

    def _resize_pair(self, img: torch.Tensor, msk: torch.Tensor,
                     size_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        img = F.interpolate(img.unsqueeze(0), size=size_hw,
                            mode="bilinear", align_corners=False).squeeze(0)
        msk = F.interpolate(msk.unsqueeze(0), size=size_hw,
                            mode="nearest").squeeze(0)
        return img, msk

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        spec = data["spec"]
        mask = data["mask"]

        # mask -> (1,H,W) float in {0,1}
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.squeeze(mask, axis=-1)
        if mask.ndim != 2:
            raise ValueError(f"Mask must be (H,W) or (H,W,1); got {mask.shape}")
        mask = (mask > 0).astype(np.float32)
        msk = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)

        # spec -> (C,H,W)
        img = self._to_chw(spec)
        img = self._match_channels(img)
        img = self._normalize(img)

        # optional resize
        if self.target_shape is not None:
            img, msk = self._resize_pair(img, msk, self.target_shape)

        return img, msk

###################################
# training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for spec, mask in dataloader:
        spec, mask = spec.to(device), mask.to(device)
        # print(spec.shape, mask.shape)
        optimizer.zero_grad() # zero the gradients 
        output = model(spec)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss = total_loss / len(dataloader)

    return loss
###################################
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    #If reduce_batch_first is False, it means we’re not treating the batch dimension as one to be reduced at this stage, 
    #so we sum only over the last two dims (-1, -2) (height, width).
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    # If input is 2D, we assume it is a single mask and reduce over the last two dimensions
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum) # Avoid division by zero
    # epsilon is added for numerical stability to avoid division by zero

    dice = (inter + epsilon) / (sets_sum + epsilon) # Similar to F1-score for pixels 2 * TP / (2*TP + FP + FN) 

    return dice.mean()

###################################
# Validation function
def validate(model, test_dataset_loader, device, criterion, threshold=0.5, visualize=False, save_dir="prediction_plots"):
    model.eval()
    total_loss = 0.0
    iou_sum = accu_sum = rec_sum = prec_sum = f1_sum = dice_score = 0.0
    # check if save_dir exists, if not create it
    if visualize and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Instantiate metrics once, on the SAME device as your tensors
    accuracy = BinaryAccuracy().to(device)
    recall = BinaryRecall().to(device)
    precision = BinaryPrecision().to(device)
    f1_score = BinaryF1Score().to(device)
    iou = BinaryJaccardIndex().to(device)

    num_val_batches = len(test_dataset_loader)

    with torch.no_grad():
        for i, (spec, mask) in enumerate(test_dataset_loader):
            # Ensure types/devices
            spec = spec.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)  # BCEWithLogitsLoss expects float targets

            # Forward + loss (logits)
            logits = model(spec)
            loss = criterion(logits, mask)
            total_loss += loss.item()

            # Binarize predictions on the SAME device
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).to(dtype=torch.float32) # Binarized predictions

            # --- metrics on tensors (no NumPy here) ---
            iou_sum  += iou(preds, mask).item()
            accu_sum += accuracy(preds, mask).item()
            rec_sum  += recall(preds, mask).item()
            prec_sum += precision(preds, mask).item()
            f1_sum   += f1_score(preds, mask).item()
            dice_score += dice_coeff(preds, mask,reduce_batch_first=False)
            # --- optional visualization (use CPU copies; do NOT overwrite variables used above) ---
            if visualize:
                spec_img = spec[0, 0].detach().cpu().numpy()
                mask_img = mask[0, 0].detach().cpu().numpy()
                pred_img = preds[0, 0].detach().cpu().numpy()

                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1); plt.title("Spectrogram")
                plt.imshow(spec_img, aspect='auto', extent=[0, 0.671, 300, 400], vmin=0, vmax=1, cmap='viridis')
                plt.xlabel("Time (s)"); plt.ylabel("Frequency (MHz)")

                plt.subplot(1, 3, 2); plt.title("True Mask")
                plt.imshow(mask_img, aspect='auto', extent=[0, 0.671, 300, 400], vmin=0, vmax=1, cmap='gray')
                plt.xlabel("Time (s)"); plt.ylabel("Frequency (MHz)")

                plt.subplot(1, 3, 3); plt.title("Predicted Mask")
                plt.imshow(pred_img, aspect='auto', extent=[0, 0.671, 300, 400], vmin=0, vmax=1, cmap='gray')
                plt.xlabel("Time (s)"); plt.ylabel("Frequency (MHz)")

                plt.tight_layout()
                plt.savefig(f"{save_dir}/predictions_testno_{i}.png")
                plt.close()

    # Averages over batches (unweighted by pixels; adjust if you want pixel-weighting)
    # max(num_val_batches, 1) is used to avoid division by zero in case num_val_batches= 0
    # Note: If you want to average over pixels instead, you can use the sum of all metrics divided by the total number of pixels
    avg_loss = total_loss / max(num_val_batches, 1)
    avg_iou = iou_sum / max(num_val_batches, 1)
    avg_accuracy = accu_sum / max(num_val_batches, 1)
    avg_recall = rec_sum / max(num_val_batches, 1)
    avg_precision = prec_sum / max(num_val_batches, 1)
    avg_F1 = f1_sum / max(num_val_batches, 1)
    avg_dice= dice_score / max(num_val_batches, 1) # Average dice score

    metrics = {
        'iou': avg_iou,
        'accuracy': avg_accuracy,
        'recall': avg_recall,
        'precision': avg_precision,
        'F1': avg_F1,
        'dice': avg_dice,
    }
    return avg_loss, metrics



if __name__ == '__main__':
    
    SEED = 123
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    n_channel=1 # 3 for RGB images
    n_class=1 # n-classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels= n_channel, n_classes=n_class, bilinear=False)  # For grayscale spectrograms, n_channels=1, n_classes=1 for binary segmentation
    model = model.to(memory_format=torch.channels_last) # to use channels last memory format for better performance
    model.to(device=device)
    learningrate=1e-3
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Loss function
    # criterion = nn.BCELoss()  # Binary Cross Entropy for pixelwise classification
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits loss for numerical stability
    # This loss function combines a sigmoid layer and the BCELoss in one single class.

    # Data
    print("Loading dataset observation data...")
    data_path = "masks_11052025" # path to the folder containing the spectrograms and masks "masks_09232025" "masks_4"
    out_channel=1 # set 3 if you use n_channels=3
    dataset= SpectrogramSegmentationDataset(
        folder=data_path,
        target_shape=None,   # or None to keep native size (1024, 1000) # (548, 747) for mask_4 folder and (1024, 1000) for mask_3 folder
        out_channels=out_channel,            # set 3 if you use n_channels=3 
        normalization=None     # "zscore", "minmax", or None
        )
    print(f"Dataset size: {len(dataset)}")
    # to split data into train/val/test sets, you can use torch.utils.data.random_split
    #  Split into train / validation partitions
    test_dataset_size = int(len(dataset) * 0.2) # 20% for validation
    train_dataset_size = len(dataset) - test_dataset_size 
    # use a fixed random seed for reproducibility
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size],generator=torch.Generator().manual_seed(SEED))
    # train_dataset_size = int(0.8 * len(dataset))
    # val_dataset_size = int(0.1 * len(dataset))
    # # test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size
    # test_dataset_size = int(0.2 * len(dataset)) 
    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, val_dataset_size, test_dataset_size])
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_dataset_size, test_dataset_size])

    # Create DataLoader
    batch_size = 4
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_dataset_loader = DataLoader(train_dataset,shuffle=True,**loader_args)
    test_dataset_loader = DataLoader(test_dataset,shuffle=False, drop_last=True, **loader_args)    

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True) # loading the spectrograms and masks

    # Training loop
    print("Starting training...")
    epochs = 68
    for epoch in range(epochs):
        loss = train(model, train_dataset_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    # Validation
    print("Starting Validation...")
    # avg_loss, metrics= validate(model, test_dataset_loader, device)
    # save_dir that matches the date and time of the run
    save_dir="prediction_plots/run_obsdata%s"%(datetime.now().strftime("%Y%m%d_%H%M"))
    binary_threshold=0.5
    avg_loss, metrics = validate(model, test_dataset_loader, device, criterion, threshold=binary_threshold, visualize=True, save_dir=save_dir)
    print("Test Performance: averaged over all test data for observation datasets")
    print(f"Avg_Loss:{avg_loss}\n dice: {metrics['dice']:.4f} \n IoU: {metrics['iou']:.4f} \n Accuracy: {metrics['accuracy']:.4f} \n"
        f" Recall: {metrics['recall']:.4f} \n Precision: {metrics['precision']:.4f} \n F1: {metrics['F1']:.4f}") 
    # print("New metrics using pytorch functions")
    # print(f"IoU: {metrics2['iou']:.4f} \n Accuracy: {metrics2['accuracy']:.4f} \n"
    #     f"Recall: {metrics2['recall']:.4f} \n Precision: {metrics2['precision']:.4f} \n F1: {metrics2['F1']:.4f}\n confusion matrix: {metrics2['Confusion_Matrix']}")

    # save the martics to a csv file
    # metrics_df = pd.DataFrame(metrics[1], index=[0]) # pd version is incompatable 
    # metrics_df.to_csv("prediction_plot/metrics_obsdata.csv", index=False)
    # Save the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_dir+"/unet_model_obsdata%s.pth"%(datetime.now().strftime("%Y%m%d_%H%M")))
    logging.info("Model saved to unet_model.pth")
    # save a log file with the model hyperparameters and performance metrics
    with open(save_dir+"/training_log_obsdata%s.txt"%(datetime.now().strftime("%Y%m%d_%H")), "w") as f:
        f.write(f"Model: UNet\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset folder: {data_path}\n")
        f.write(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        f.write(f"Normalization: None\n")
        f.write(f"Output channels: {out_channel}\n")
        f.write(f"Input channels: {n_channel}\n")
        f.write(f"Threshold: {binary_threshold}\n")
        f.write(f"Train dataset size: {len(train_dataset)}\n")
        f.write(f"Test dataset size: {len(test_dataset)}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learningrate}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss function: BCEWithLogitsLoss\n")
        f.write(f"Number of epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write("Performance on test dataset:\n")
        f.write(f"Avg_Loss: {avg_loss}\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    logging.info("Training log saved to training_log.txt")

    # Save the model architecture
    # with open("unet_model_architecture.txt", "w") as f:
    #     f.write(str(model))
    # logging.info("Model architecture saved to unet_model_architecture.txt")
    # Save the model weights
    # torch.save(model.state_dict(), "prediction_plots/unet_model_weights.pth")
    # logging.info("Model weights saved to unet_model_weights.pth")
    # # Save the model optimizer state
    # torch.save(optimizer.state_dict(), "prediction_plots/unet_optimizer_state.pth")
    # logging.info("Optimizer state saved to unet_optimizer_state.pth")
