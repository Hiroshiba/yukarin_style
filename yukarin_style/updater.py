from pytorch_trainer.dataset import convert
from pytorch_trainer.training import StandardUpdater
import torch


class Updater(StandardUpdater):
    def __init__(
        self, iterator, optimizer, model, moving_average_rate: float, device=None,
    ):
        super().__init__(
            iterator, optimizer, model, device=device,
        )

        self.moving_average_rate = moving_average_rate

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        style_transfer_optimizer = self._optimizers["style_transfer"]
        mapping_network_optimizer = self._optimizers["mapping_network"]
        style_encoder_optimizer = self._optimizers["style_encoder"]
        discriminator_optimizer = self._optimizers["discriminator"]

        generator_model = self._models["generator"]
        discriminator_model = self._models["discriminator"]

        for m in self._models.values():
            m.train()

        # discriminator
        loss = discriminator_model.forward_with_latent(**in_arrays)
        discriminator_optimizer.zero_grad()
        loss.backward()
        discriminator_optimizer.step()

        loss = discriminator_model.forward_with_reference(**in_arrays)
        discriminator_optimizer.zero_grad()
        loss.backward()
        discriminator_optimizer.step()

        # predictor
        loss = generator_model.forward_with_latent(**in_arrays)
        style_transfer_optimizer.zero_grad()
        mapping_network_optimizer.zero_grad()
        style_encoder_optimizer.zero_grad()
        loss.backward()
        style_transfer_optimizer.step()
        mapping_network_optimizer.step()
        style_encoder_optimizer.step()

        loss = generator_model.forward_with_reference(**in_arrays)
        style_transfer_optimizer.zero_grad()
        loss.backward()
        style_transfer_optimizer.step()

        # moving average
        moving_generator_model = self._models["moving_generator"]
        for param1, param2 in zip(
            generator_model.parameters(), moving_generator_model.parameters()
        ):
            param2.data = torch.lerp(param1.data, param2.data, self.moving_average_rate)
