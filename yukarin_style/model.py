from typing import NamedTuple, Optional
from pytorch_trainer import report
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from yukarin_style.config import ModelConfig, NetworkConfig
from yukarin_style.network.discriminator import Discriminator
from yukarin_style.network.mapping_network import MappingNetwork, create_mapping_network
from yukarin_style.network.style_encoder import StyleEncoder, create_style_encoder
from yukarin_style.network.style_transfer import StyleTransfer, create_style_transfer


class Networks(NamedTuple):
    style_transfer: StyleTransfer
    mapping_network: MappingNetwork
    style_encoder: StyleEncoder
    discriminator: Discriminator


def create_network(config: NetworkConfig):
    return Networks(
        style_transfer=create_style_transfer(config),
        mapping_network=create_mapping_network(config),
        style_encoder=create_style_encoder(config),
        discriminator=Discriminator(
            input_size=config.feature_size,
            min_hidden_size=config.discriminator.min_hidden_size,
            max_hidden_size=config.discriminator.max_hidden_size,
            kernel_size=config.discriminator.kernel_size,
            residual_block_num=config.discriminator.residual_block_num,
            last_kernel_size=config.discriminator.last_kernel_size,
        ),
    )


def calc_adversarial_loss(x: Tensor, is_real: bool):
    if is_real:
        t = torch.ones_like(x)
    else:
        t = torch.zeros_like(x)
    return F.binary_cross_entropy_with_logits(x, t)


def calc_r1_loss(output, input):
    grad = torch.autograd.grad(
        outputs=output.sum(),
        inputs=input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    reg = 0.5 * grad.pow(2).reshape(input.shape[0], -1).sum(1).mean()
    return reg


class GeneratorModel(nn.Module):
    def __init__(self, model_config: ModelConfig, networks: Networks) -> None:
        super().__init__()
        self.model_config = model_config
        self.style_transfer = networks.style_transfer
        self.mapping_network = networks.mapping_network
        self.style_encoder = networks.style_encoder
        self.discriminator = networks.discriminator

    def forward(
        self,
        x: Tensor,
        x_ref1: Optional[Tensor],
        x_ref2: Optional[Tensor],
        z1: Optional[Tensor],
        z2: Optional[Tensor],
        prefix: str,
    ):
        assert (x_ref1 is None) != (z1 is None)

        pad = self.model_config.padding_length

        if z1 is not None:
            s1 = self.mapping_network(z1)
            s2 = self.mapping_network(z2)
        else:
            s1 = self.style_encoder(x_ref1)
            s2 = self.style_encoder(x_ref2)

        # adversarial loss
        y1 = self.style_transfer(x=x, s=s1)
        loss_adv = calc_adversarial_loss(
            x=self.discriminator(y1[:, pad:-pad]), is_real=True
        )

        # style reconstruction loss
        s1_re = self.style_encoder(y1[:, pad:-pad])
        loss_style = torch.mean(torch.abs(s1_re - s1))

        # diversity sensitive loss
        y2 = self.style_transfer(x=x, s=s2)
        y2 = y2.detach()
        loss_diverse = -torch.mean(torch.abs(y1[:, pad:-pad] - y2[:, pad:-pad]))

        # cycle-consistency loss
        s_x = self.style_encoder(x[:, pad * 2 : -pad * 2])
        x_re = self.style_transfer(y1, s_x)
        loss_cycle = torch.mean(torch.abs(x_re - x[:, pad * 2 : -pad * 2]))

        # identification loss
        x_id = self.style_transfer(x, s_x)
        loss_identify = torch.mean(
            torch.abs(x_id[:, pad:-pad] - x[:, pad * 2 : -pad * 2])
        )

        loss = (
            loss_adv
            + self.model_config.style_reconstruction_weight * loss_style
            + self.model_config.diversity_sensitive_weight * loss_diverse
            + self.model_config.cycle_consistency_weight * loss_cycle
            + self.model_config.identification_weight * loss_identify
        )

        # report
        values = {
            f"{prefix}/loss": loss,
            f"{prefix}/loss_adv": loss_adv,
            f"{prefix}/loss_style": loss_style,
            f"{prefix}/loss_diverse": loss_diverse,
            f"{prefix}/loss_cycle": loss_cycle,
            f"{prefix}/loss_identify": loss_identify,
        }
        if not self.training:
            weight = x.shape[0]
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss

    def forward_with_latent(
        self,
        x: Tensor,
        x_ref1: Optional[Tensor],
        x_ref2: Optional[Tensor],
        z1: Optional[Tensor],
        z2: Optional[Tensor],
    ):
        return self(x=x, x_ref1=None, x_ref2=None, z1=z1, z2=z2, prefix="latent")

    def forward_with_reference(
        self,
        x: Tensor,
        x_ref1: Optional[Tensor],
        x_ref2: Optional[Tensor],
        z1: Optional[Tensor],
        z2: Optional[Tensor],
    ):
        return self(x=x, x_ref1=x_ref1, x_ref2=x_ref2, z1=None, z2=None, prefix="ref")


class DiscriminatorModel(nn.Module):
    def __init__(self, model_config: ModelConfig, networks: Networks) -> None:
        super().__init__()
        self.model_config = model_config
        self.style_transfer = networks.style_transfer
        self.mapping_network = networks.mapping_network
        self.style_encoder = networks.style_encoder
        self.discriminator = networks.discriminator

    def forward(
        self, x: Tensor, x_ref: Optional[Tensor], z: Optional[Tensor], prefix: str,
    ):
        assert (x_ref is None) != (z is None)

        pad = self.model_config.padding_length

        # r1 loss
        with torch.enable_grad():
            x_r1 = x[:, pad * 2 : -pad * 2]
            x_r1.requires_grad_()
            real = self.discriminator(x_r1)
            loss_r1 = calc_r1_loss(output=real, input=x_r1)

        # real loss
        loss_real = calc_adversarial_loss(x=real, is_real=True)

        # fake loss
        with torch.no_grad():
            if z is not None:
                s = self.mapping_network(z)
            else:
                s = self.style_encoder(x_ref)

            y = self.style_transfer(x=x, s=s)
        loss_fake = calc_adversarial_loss(
            x=self.discriminator(y[:, pad:-pad]), is_real=False
        )

        loss = loss_real + loss_fake + self.model_config.r1_weight * loss_r1

        # report
        values = {
            f"{prefix}/loss": loss,
            f"{prefix}/loss_real": loss_real,
            f"{prefix}/loss_fake": loss_fake,
            f"{prefix}/loss_r1": loss_r1,
        }
        if not self.training:
            weight = x.shape[0]
            values = {key: (l, weight) for key, l in values.items()}  # add weight
        report(values, self)

        return loss

    def forward_with_latent(
        self,
        x: Tensor,
        x_ref1: Optional[Tensor],
        x_ref2: Optional[Tensor],
        z1: Optional[Tensor],
        z2: Optional[Tensor],
    ):
        return self(x=x, x_ref=None, z=z1, prefix="latent")

    def forward_with_reference(
        self,
        x: Tensor,
        x_ref1: Optional[Tensor],
        x_ref2: Optional[Tensor],
        z1: Optional[Tensor],
        z2: Optional[Tensor],
    ):
        return self(x=x, x_ref=x_ref1, z=None, prefix="ref")
