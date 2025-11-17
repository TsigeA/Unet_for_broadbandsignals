
"""A simple U-Net implementation for spectrogram segmentation using PyTorch.
This script defines a U-Net model, trains it on observation data from GBT, and evaluates its performance.
It includes:
- A ConvBlock class for building convolutional blocks.
- A SimpleUNet class for the U-Net architecture.
- A training function to train the model.
- A dataset class to load spectrograms and masks from files.
- A validation function to evaluate the model's performance.
- training data (mask) contains spectrograms and corresponding masks of only the broadband signals
Spectrograms without broadband signals will have masks of all zeros.
@author: Tsige Atilaw
@date: 2025-07-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
# import pandas as pd
import glob
import cv2  # For image processing
# from scipy.ndimage import zoom  # For resizing
from unet import UNet
import logging
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryJaccardIndex,
)
# to do the metric calculations using the built-in pytorch functions

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

class SpectrogramSegmentationDataset(Dataset):
    def __init__(self, folder, target_shape=(1024, 1000)):
        self.files = sorted(glob.glob(f"{folder}/*.npz"))
        self.target_shape = target_shape  # Target shape for resizing

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # spec = (data["spec"] - np.mean(data["spec"])) / np.std(data["spec"])  # z-normalize
        # make sure the spectrogram is in two dimensions (x,y)
        if len(data["spec"].shape) == 3:  # If the spectrogram has a channel dimension
            spec = data["spec"][:,:, 0]  # Take the first channel (assuming it's a single-channel spectrogram)  
        else:
            spec = data["spec"]  # Use the original spectrogram without normalization
        
        mask = data["mask"]
        # use resizing only for dataset in mask_3 folder
        # datasets in mask_4 folder have spectrograms of same size     
        # Resize spectrogram and mask to the target shape or the same size
        # spec = cv2.resize(spec, (1024, 1000), interpolation=cv2.INTER_CUBIC)
        # mask = cv2.resize(mask, (1024, 1000), interpolation=cv2.INTER_CUBIC)
        # spec = self.resize_array(spec, self.target_shape)
        # mask = self.resize_array(mask, self.target_shape)

        # Convert to PyTorch tensors
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return spec, mask

    # def resize_array(self, array, target_shape):
    #     # Resize using numpy's slicing and padding
    #     resized = np.zeros(target_shape, dtype=array.dtype)
    #     min_shape = (
    #         min(target_shape[0], array.shape[0]),
    #         min(target_shape[1], array.shape[1]),
    #     )
    #     resized[:min_shape[0], :min_shape[1]] = array[:min_shape[0], :min_shape[1]]
    #     return resized
def validate(model, test_dataset_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
   
    
    with torch.no_grad():
        i= 0
        for spec, mask in test_dataset_loader:
            spec, mask = spec.to(device), mask.to(device)
            # output = torch.sigmoid(model(spec))
            # loss = criterion(output, mask)
            output = model(spec) 
            loss = criterion(output, mask) # since criterion is BCEWithLogitsLoss, we don't need to apply sigmoid here
            total_loss += loss.item()

            all_preds.append(output.cpu())
            all_labels.append(mask.cpu())
            # to visualize the predictions
            spec = spec[0, 0].cpu().numpy()  # Move tensor to CPU before converting to NumPy
            mask = mask[0, 0].cpu().numpy()  # Move tensor to CPU before converting to NumPy
            pred = output[0, 0].cpu().numpy()  # Move tensor to CPU before converting to NumPy
            # Plot the spectrogram, true mask, and predicted mask
            plt.figure(figsize=(15, 4))
            plt.subplot(1, 3, 1)
            plt.title("Spectrogram")
            plt.imshow(spec, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0, cmap='viridis')
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (MHz)")
            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(mask, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0, cmap='gray')
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (MHz)")
            plt.subplot(1, 3, 3)
            plt.title("Predicted Mask")
            plt.imshow(pred > 0.5, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0,cmap='gray')  # threshold
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (MHz)")
            plt.tight_layout()
            plt.savefig(f"prediction_plots/predictions_testno_{i}.png")
            plt.close()
            i += 1

    avg_loss = total_loss / len(test_dataset_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = evaluate_predictions(all_preds, all_labels) # calculate the metrics
    metrics2 = evaluate_predictions_pytorchfun(all_preds, all_labels) # calculate the metrics
    return avg_loss, metrics,metrics2

def evaluate_predictions(preds, labels, threshold=0.5): # the threshold is used to convert logits/probabilities to binary predictions
    """
    preds, labels: tensors of shape [B, 1, H, W]
    """
    preds = (preds > threshold).float()
    labels = labels.float()

    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = (preds + labels).clamp(0, 1).sum(dim=(1, 2, 3))
    union= union - intersection  # Union is sum of both masks minus intersection
    dice = (2. * intersection + 1e-6) / (preds.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) + 1e-6)# Similar to F1-score for pixels 2 * TP / (2*TP + FP + FN)
    iou = (intersection + 1e-6) / (union + 1e-6) # How much predicted mask overlaps with truth	intersection / union
    # acc = (preds == labels).float().mean(dim=(1, 2, 3)) # How many pixels are correctly classified	(pred == true).sum() / total
    xor= (preds == labels).float().sum(dim=(1, 2, 3)) # XOR operation to find the number of pixels that are different
    acc = xor/ (union + xor- intersection) # Accuracy is the ratio of correctly predicted pixels to total pixels and it gives high value in image segmentation b/c of the TN (backgroad) is dominant 
    recall = intersection / (labels.sum(dim=(1, 2, 3)) + 1e-6) # How many true positives were found TP / (TP + FN)
    precision = intersection / (preds.sum(dim=(1, 2, 3)) + 1e-6) # How many predicted positives were true TP / (TP + FP)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-6) # F1 score
    metrics = {
        'dice': dice.mean().item(),
        'iou': iou.mean().item(),
        'accuracy': acc.mean().item(), 
        'recall': recall.mean().item(),
        'precision': precision.mean().item(),
        'F1': F1.mean().item()
    }
    return metrics

def evaluate_predictions_pytorchfun(preds, labels, threshold=0.5):
    """
    Evaluate predictions using PyTorch metric functions.
    Args:
        preds: Tensor of shape [B, 1, H, W] - predicted logits/probabilities.
        labels: Tensor of shape [B, 1, H, W] - ground truth binary masks.
        threshold: Float - threshold to convert logits/probabilities to binary predictions.
    Returns:
        Dictionary containing metrics: Accuracy, Recall, Precision, F1.
    """
    preds = (preds > threshold).float()
    labels = labels.float()

    # Flatten tensors for metric calculations
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    # Instantiate metrics
    accuracy = BinaryAccuracy()
    recall = BinaryRecall()
    precision = BinaryPrecision()
    f1_score = BinaryF1Score()
    iou = BinaryJaccardIndex()
    # dice = BinaryDice()
    confusion_matrix = BinaryConfusionMatrix()# num_classes=2 for binary segmentation

    metrics2 = {
        'iou': iou(preds_flat, labels_flat).item(),
        # 'Dice': dice(preds_flat, labels_flat).item(),
        'accuracy': accuracy(preds_flat, labels_flat).item(),
        'recall': recall(preds_flat, labels_flat).item(),
        'precision': precision(preds_flat, labels_flat).item(),
        'F1': f1_score(preds_flat, labels_flat).item(),
        'Confusion_Matrix': confusion_matrix(preds_flat, labels_flat).cpu().numpy()
    }

    return metrics2

if __name__ == '__main__':
    
    SEED = 42
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
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=1, bilinear=False)  # For grayscale spectrograms, n_channels=1, n_classes=1 for binary segmentation
    model = model.to(memory_format=torch.channels_last) # to use channels last memory format for better performance

    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Loss function
    # criterion = nn.BCELoss()  # Binary Cross Entropy for pixelwise classification
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits loss for numerical stability
    # This loss function combines a sigmoid layer and the BCELoss in one single class.


    # Data
    print("Loading dataset observation data...")
    dataset = SpectrogramSegmentationDataset("masks_4")
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
    for epoch in range(100):
        loss = train(model, train_dataset_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    # Validation
    print("Starting Validation...")
    avg_loss, metrics,metrics2= validate(model, test_dataset_loader, device)
    print("Test Performance: averaged over all test data for observation datasets")

    print(f"Avg_Loss:{avg_loss}\n dice: {metrics['dice']:.4f} \n IoU: {metrics['iou']:.4f} \n Accuracy: {metrics['accuracy']:.4f} \n"
        f" Recall: {metrics['recall']:.4f} \n Precision: {metrics['precision']:.4f} \n F1: {metrics['F1']:.4f}") 
    print("New metrics using pytorch functions")
    print(f"IoU: {metrics2['iou']:.4f} \n Accuracy: {metrics2['accuracy']:.4f} \n"
        f"Recall: {metrics2['recall']:.4f} \n Precision: {metrics2['precision']:.4f} \n F1: {metrics2['F1']:.4f}\n confusion matrix: {metrics2['Confusion_Matrix']}")

    # save the martics to a csv file
    # metrics_df = pd.DataFrame(metrics[1], index=[0]) # pd version is incompatable 
    # metrics_df.to_csv("prediction_plot/metrics_obsdata.csv", index=False)
    # Save the model
    torch.save(model.state_dict(), "prediction_plots/unet_model_obsdata.pth")
    logging.info("Model saved to unet_model.pth")
    # Save the model architecture
    with open("unet_model_architecture.txt", "w") as f:
        f.write(str(model))
    logging.info("Model architecture saved to unet_model_architecture.txt")
    # Save the model weights
    # torch.save(model.state_dict(), "prediction_plots/unet_model_weights.pth")
    # logging.info("Model weights saved to unet_model_weights.pth")
    # # Save the model optimizer state
    # torch.save(optimizer.state_dict(), "prediction_plots/unet_optimizer_state.pth")
    # logging.info("Optimizer state saved to unet_optimizer_state.pth")
