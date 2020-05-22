from torch import Tensor, nn

from yukarin_style.network.adaptive_instance_normalization import AdaptiveInstanceNorm1d
import math


class ResidualBlock(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, kernel_size: int, downsample: bool
    ):
        super().__init__()

        self.norm1 = nn.InstanceNorm1d(num_features=input_size, affine=True)
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        self.residual_pool = nn.AvgPool1d(kernel_size=2) if downsample else None

        self.norm2 = nn.InstanceNorm1d(num_features=input_size, affine=True)
        self.activation2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        self.shortcut_conv = (
            nn.Conv1d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            if input_size != output_size
            else None
        )
        self.shortcut_pool = nn.AvgPool1d(kernel_size=2) if downsample else None

    def forward(self, x: Tensor):
        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        if self.residual_pool is not None:
            h = self.residual_pool(h)

        h = self.norm2(h)
        h = self.activation2(h)
        h = self.conv2(h)

        if self.shortcut_conv is not None:
            x = self.shortcut_conv(x)

        if self.shortcut_pool is not None:
            x = self.shortcut_pool(x)

        return (x + h) / math.sqrt(2)


class AdaptiveResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, style_size: int, kernel_size: int):
        super().__init__()

        self.norm1 = AdaptiveInstanceNorm1d(
            feature_size=hidden_size, style_size=style_size
        )
        self.activation1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        self.norm2 = AdaptiveInstanceNorm1d(
            feature_size=hidden_size, style_size=style_size
        )
        self.activation2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor, s: Tensor):
        h = self.norm1(x=x, s=s)
        h = self.activation1(h)
        h = self.conv1(h)

        h = self.norm2(x=h, s=s)
        h = self.activation2(h)
        h = self.conv2(h)
        return (x + h) / math.sqrt(2)
