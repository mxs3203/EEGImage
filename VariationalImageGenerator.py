import torch.nn as nn
import torch
import torch.nn.functional as F

class ImageGenerator(nn.Module):
    def __init__(self,  device, image_channels=1, input_size=256, scale_factor=7):
        super(ImageGenerator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.image_channels = image_channels
        self.scale_factor = scale_factor
        self.fc1 = nn.Linear(input_size, 256 * scale_factor * scale_factor)
        self.fc2 = nn.Linear(256 * scale_factor * scale_factor, 256 * scale_factor * scale_factor)
        self.a = nn.ReLU()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.a(self.fc1(x))
        x = self.a(self.fc2(x))
        x = x.view(-1, 256, self.scale_factor, self.scale_factor)  # Reshape to 4D tensor
        x = self.deconv_layers(x)
        return x
