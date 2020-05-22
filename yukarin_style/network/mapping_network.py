from torch import Tensor, nn


class MappingNetwork(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, layer_num: int
    ):
        super().__init__()

        layers = []
        for i_layer in range(layer_num):
            is_first = i_layer == 0
            is_last = i_layer == layer_num - 1
            layers.append(
                nn.Linear(
                    in_features=input_size if is_first else hidden_size,
                    out_features=output_size if is_last else hidden_size,
                )
            )

            if not is_last:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        return self.layers(x)
