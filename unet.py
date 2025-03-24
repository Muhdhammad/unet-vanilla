import torch
import torch.nn as nn

class doubleConv(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.double_conv(x)

class downSample(nn.Module):

  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = doubleConv(in_channels, out_channels)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    down = self.conv(x)
    p = self.pool(down)

    return down, p

class upSample(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
    self.conv = doubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1)
    x = torch.cat([x1, x2], 1)
    return self.conv(x)

class unet(nn.Module):
  def __init__(self, in_channels, num_classes):
    super().__init__()

    self.down_conv1 = downSample(in_channels, 64)
    self.down_conv2 = downSample(64, 128)
    self.down_conv3 = downSample(128, 256)
    self.down_conv4 = downSample(256, 512)

    self.bottleneck = doubleConv(512, 1024)

    self.up_conv1 = upSample(1024, 512)
    self.up_conv2 = upSample(512, 256)
    self.up_conv3 = upSample(256, 128)
    self.up_conv4 = upSample(128, 64)

    self.out = nn.Conv2d(in_channels=64 , out_channels=num_classes, kernel_size=1)

  def forward(self, x):

    down1, p1 = self.down_conv1(x)
    down2, p2 = self.down_conv2(p1)
    down3, p3 = self.down_conv3(p2)
    down4, p4 = self.down_conv4(p3)

    b = self.bottleneck(p4)

    up1 = self.up_conv1(b, down4)
    up2 = self.up_conv2(up1, down3)
    up3 = self.up_conv3(up2, down2)
    up4 = self.up_conv4(up3, down1)

    output = self.out(up4)
    return output

input_image = torch.rand(1, 1, 512, 512)

model = unet(1, 2)
output = model(input_image)
print(output.size())

