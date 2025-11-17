"""Using a Unet model to segment images.

@author: Tsige Atilaw
@date: 2025-07-03
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
    BinaryJaccardIndex,
)

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
    def __init__(self, folder):
        self.files = sorted(glob.glob(f"{folder}/*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # spec = (data["spec"] - np.mean(data["spec"])) / np.std(data["spec"])  # z-normalize
        # spec = np.log1p(data["spec"])  # log normalization
        spec = (data["spec"] - np.min(data["spec"])) / (np.max(data["spec"]) - np.min(data["spec"]))  # min-max normalization
        mask = data["mask"]

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return spec, mask
# inference and visualization
def evaluate_predictions(preds, labels, threshold=0.5): # the threshold is used to convert logits/probabilities to binary predictions
    """
    preds, labels: tensors of shape [B, 1, H, W] - predicted logits/probabilities and ground truth binary masks.
    """
    preds = (preds > threshold).float()
    labels = labels.float()
    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)

    # intersection = (preds * labels).sum(dim=(1, 2, 3))
    intersection = (preds * labels).sum()
    total_sum= (preds + labels).sum() # Total sum of predicted and true masks
    # union = (preds + labels).clamp(0, 1).sum(dim=(1, 2, 3))
    union= preds.sum()+labels.sum() - intersection  # Union is sum of both masks minus intersection
    # dice = (2. * intersection + 1e-6) / (preds.sum(dim=(1,2,3)) + labels.sum(dim=(1,2,3)) + 1e-6)# Similar to F1-score for pixels 2 * TP / (2*TP + FP + FN)
    dice= (2. * intersection) / (total_sum) # Similar to F1-score for pixels 2 * TP / (2*TP + FP + FN)
    iou = (intersection/union) # How much predicted mask overlaps with truth	intersection / union
    # acc = (preds == labels).float().mean(dim=(1, 2, 3)) # How many pixels are correctly classified	(pred == true).sum() / total
    xor= (preds == labels).float().sum() # XOR operation to find the number of pixels that are different
    acc = xor/ (union + xor- intersection) # Accuracy is the ratio of correctly predicted pixels to total pixels and it gives high value in image segmentation b/c of the TN (backgroad) is dominant 
    recall = intersection / (labels.sum()) # How many true positives were found TP / (TP + FN)
    precision = intersection / (preds.sum()) # How many predicted positives were true TP / (TP + FP)
    F1 = 2 * (precision * recall) / (precision + recall) # F1 score
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
            # plt.figure(figsize=(15, 4))
            # plt.subplot(1, 3, 1)
            # plt.title("Spectrogram")
            # plt.imshow(spec, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0, cmap='viridis')
            # plt.xlabel("Time (s)")
            # plt.ylabel("Frequency (MHz)")
            # plt.subplot(1, 3, 2)
            # plt.title("True Mask")
            # plt.imshow(mask, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0, cmap='gray')
            # plt.xlabel("Time (s)")
            # plt.ylabel("Frequency (MHz)")
            # plt.subplot(1, 3, 3)
            # plt.title("Predicted Mask")
            # plt.imshow(pred > 0.5, aspect='auto',extent=[0, 0.671, 300, 400],vmax=1,vmin=0,cmap='gray')  # threshold
            # plt.xlabel("Time (s)")
            # plt.ylabel("Frequency (MHz)")
            # plt.tight_layout()
            # plt.savefig(f"prediction_plots/predictions_testno_{i}.png")
            # plt.close()
            i += 1

    avg_loss = total_loss / len(test_dataset_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = evaluate_predictions(all_preds, all_labels) # calculate the metrics
    metrics2 = evaluate_predictions_pytorchfun(all_preds, all_labels) # calculate the metrics
    return avg_loss, metrics, metrics2

# training excusion
# Settings
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# model = SimpleUNet().to(device)
# model.summary()  # print the model summary
# Print model summary
# summary(model, input_size=(1, 1024, 1000))  # Input size for grayscale spectrograms

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
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)) 

    # Loss function
    # criterion = nn.BCELoss()  # Binary Cross Entropy for pixelwise classification
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy with logits loss for numerical stability
    # This loss function combines a sigmoid layer and the BCELoss in one single class.

    # Data
    print("Loading dataset...")
    dataset = SpectrogramSegmentationDataset("synthetic_spectrograms_4") 
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
    print("Starting training of synthetic data...")
    for epoch in range(5):
        loss = train(model, train_dataset_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    # Validation
    print("Starting Validation...")
    avg_loss, metrics, metrics2 = validate(model, test_dataset_loader, device)
    print("Test Performance: averaged over all test data")
    print(f"Avg_Loss:{avg_loss}\n dice: {metrics['dice']:.4f} \n IoU: {metrics['iou']:.4f} \n Accuracy: {metrics['accuracy']:.4f} \n"
        f" Recall: {metrics['recall']:.4f} \n Precision: {metrics['precision']:.4f} \n F1: {metrics['F1']:.4f}") 
    print("New metrics using pytorch functions")
    print(f"IoU: {metrics2['iou']:.4f} \n Accuracy: {metrics2['accuracy']:.4f} \n"
        f"Recall: {metrics2['recall']:.4f} \n Precision: {metrics2['precision']:.4f} \n F1: {metrics2['F1']:.4f}\n confusion matrix: {metrics2['Confusion_Matrix']}")

    # save the martics to a csv file
    # metrics_df = pd.DataFrame(metrics[1], index=[0])
    # metrics_df.to_csv("prediction_plot/metrics_syntheticdata.csv", index=False)
    # Save the model
    # torch.save(model.state_dict(), "prediction_plots/unet_synthetic_spec.pth")
    # print("Model saved to unet_synthetic_spec.pth")





