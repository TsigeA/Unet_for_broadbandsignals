""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    """
    A U-Net architecture for semantic segmentation tasks.

    Args:
        n_channels (int): Number of input channels.
        n_classes (int): Number of output classes. n_classes=1 for binary segmentation
        bilinear (bool, optional): If True, use bilinear upsampling. Default is False.

    Attributes:
        inc (DoubleConv): Initial convolution block.
        down1, down2, down3, down4 (Down): Downsampling blocks.
        up1, up2, up3, up4 (Up): Upsampling blocks.
        outc (OutConv): Final output convolution layer.

    Methods:
        forward(x): Forward pass of the network.
        use_checkpointing(): Enables gradient checkpointing for memory efficiency.
    """
    def __init__(self, n_channels, n_classes, bilinear=False):
        # Encoder outputs (x1 to x4) are skip-connected into the decoder path (up4 to up1).
        # These skip connections help preserve spatial information lost during downsampling
        # and improve the reconstruction of fine details in the output.
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        'Encoder outputs (x1 to x4) are skip-connected into the decoder path (up4 to up1)'
        # x is the input tensor with shape (batch_size, n_channels, height, width)
        x1 = self.inc(x) # Initial convolution
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Final output layer
        # logits will have shape (batch_size, n_classes, height, width)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)