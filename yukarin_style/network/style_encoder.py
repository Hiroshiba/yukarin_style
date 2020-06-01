from yukarin_style.network.downsample_network import DoensampleNetwork

from yukarin_style.config import NetworkConfig


class StyleEncoder(DoensampleNetwork):
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
        super().__init__(
            input_size=input_size,
            min_hidden_size=min_hidden_size,
            max_hidden_size=max_hidden_size,
            kernel_size=kernel_size,
            output_size=output_size,
            residual_block_num=residual_block_num,
            last_kernel_size=last_kernel_size,
        )


def create_style_encoder(config: NetworkConfig):
    return StyleEncoder(
        input_size=config.feature_size,
        min_hidden_size=config.style_encoder.min_hidden_size,
        max_hidden_size=config.style_encoder.max_hidden_size,
        kernel_size=config.style_encoder.kernel_size,
        output_size=config.style_size,
        residual_block_num=config.style_encoder.residual_block_num,
        last_kernel_size=config.style_encoder.last_kernel_size,
    )
