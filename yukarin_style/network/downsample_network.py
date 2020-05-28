from torch import Tensor, nn

from yukarin_style.network.residual_block import ResidualBlock


class DoensampleNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        min_hidden_size: int,
        max_hidden_size: int,
        kernel_size: int,
        output_size: int,
        residual_block_num: int,
        last_kernel_size: int,
    ):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=min_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        for i_block in range(residual_block_num):
            s1 = min_hidden_size * 2 ** i_block
            s1 = s1 if s1 <= max_hidden_size else max_hidden_size

            s2 = min_hidden_size * 2 ** (i_block + 1)
            s2 = s2 if s2 <= max_hidden_size else max_hidden_size

            layers.append(
                ResidualBlock(
                    input_size=s1,
                    output_size=s2,
                    kernel_size=kernel_size,
                    downsample=True,
                )
            )

        s = min_hidden_size * 2 ** residual_block_num
        s = s if s <= max_hidden_size else max_hidden_size

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(
            nn.Conv1d(
                in_channels=s,
                out_channels=s,
                kernel_size=last_kernel_size,
                stride=1,
                padding=0,
            )
        )
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

        self.tail = nn.Linear(in_features=s, out_features=output_size)

    def forward(self, x: Tensor):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.squeeze(2)
        x = self.tail(x)
        return x
