from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from yukarin_style.config import Config
from yukarin_style.dataset import generate_latent
from yukarin_style.network.mapping_network import MappingNetwork, create_mapping_network
from yukarin_style.network.style_encoder import StyleEncoder, create_style_encoder
from yukarin_style.network.style_transfer import StyleTransfer, create_style_transfer


class Generator(object):
    def __init__(
        self,
        config: Config,
        style_transfer: Union[StyleTransfer, Path],
        mapping_network: Union[MappingNetwork, Path],
        style_encoder: Union[StyleEncoder, Path],
        use_gpu: bool,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(style_transfer, Path):
            state_dict = torch.load(style_transfer)
            style_transfer = create_style_transfer(config.network)
            style_transfer.load_state_dict(state_dict)

        self.style_transfer = style_transfer.eval().to(self.device)

        if isinstance(mapping_network, Path):
            state_dict = torch.load(mapping_network)
            mapping_network = create_mapping_network(config.network)
            mapping_network.load_state_dict(state_dict)

        self.mapping_network = mapping_network.eval().to(self.device)

        if isinstance(style_encoder, Path):
            state_dict = torch.load(style_encoder)
            style_encoder = create_style_encoder(config.network)
            style_encoder.load_state_dict(state_dict)

        self.style_encoder = style_encoder.eval().to(self.device)

    def generate_latent(self, batch_size: int):
        return torch.stack(
            [
                torch.from_numpy(generate_latent(self.config.network.latent_size))
                for _ in range(batch_size)
            ]
        )

    def generate_style(
        self, x: Optional[Tensor], z: Optional[Tensor],
    ):
        assert (x is None) != (z is None)
        if x is not None:
            with torch.no_grad():
                x = x.to(self.device)
                return self.style_encoder(x)
        else:
            with torch.no_grad():
                z = z.to(self.device)
                return self.mapping_network(z)

    def generate(
        self, x: Tensor, s: Tensor,
    ):
        x = x.to(self.device)
        s = s.to(self.device)
        with torch.no_grad():
            y = self.style_transfer(x=x, s=s)
        return y.cpu().numpy()
