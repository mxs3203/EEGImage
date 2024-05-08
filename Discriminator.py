
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self, image_size=28):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(64 * (image_size // 4) * (image_size // 4), 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 64 * (self.image_size // 4) * (self.image_size // 4))
        x = self.fc(x)
        return x