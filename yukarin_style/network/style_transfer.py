from yukarin_style.network.residual_block import AdaptiveResidualBlock, ResidualBlock
from torch import Tensor, nn

from yukarin_style.config import NetworkConfig


class StyleTransfer(nn.Module):
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        style_size: int,
        kernel_size: int,
        residual_block_num: int,
        adaptive_residual_block_num: int,
    ):
        super().__init__()

        self.head = nn.Conv1d(
            in_channels=feature_size,
            out_channels=hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    input_size=hidden_size,
                    output_size=hidden_size,
                    kernel_size=kernel_size,
                    downsample=False,
                )
                for _ in range(residual_block_num)
            ]
        )

        self.adaptive_residual_blocks = nn.ModuleList(
            [
                AdaptiveResidualBlock(
                    hidden_size=hidden_size,
                    style_size=style_size,
                    kernel_size=kernel_size,
                )
                for _ in range(adaptive_residual_block_num)
            ]
        )

        self.tail = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=feature_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor, s: Tensor):
        x = self.head(x)

        x = self.residual_blocks(x)

        for layer in self.adaptive_residual_blocks:
            x = layer(x=x, s=s)

        x = self.tail(x)
        return x


def create_style_transfer(config: NetworkConfig):
    return StyleTransfer(
        feature_size=config.feature_size,
        hidden_size=config.style_transfer.hidden_size,
        style_size=config.style_size,
        kernel_size=config.style_transfer.kernel_size,
        residual_block_num=config.style_transfer.residual_block_num,
        adaptive_residual_block_num=config.style_transfer.adaptive_residual_block_num,
    )
