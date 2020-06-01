import warnings
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from pytorch_trainer.iterators import MultiprocessIterator
from pytorch_trainer.training import extensions, Trainer
from tensorboardX import SummaryWriter
from torch import optim, nn

from yukarin_style.config import Config
from yukarin_style.dataset import create_dataset
from yukarin_style.model import GeneratorModel, DiscriminatorModel, create_network
from yukarin_style.utility.tensorboard_extension import TensorboardReport
from yukarin_style.updater import Updater
from yukarin_style.utility.chainer_utility import ObjectLinearShift


def create_optimizer(config: Dict[str, Any], model: nn.Module):
    cp: Dict[str, Any] = copy(config)
    n = cp.pop("name").lower()

    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    return optimizer


def create_trainer(
    config_dict: Dict[str, Any], output: Path,
):
    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # model
    device = torch.device("cuda")
    networks = create_network(config.network)
    generator_model = GeneratorModel(model_config=config.model, networks=networks).to(
        device
    )
    moving_generator_model = deepcopy(generator_model).to(device)
    discriminator_model = DiscriminatorModel(
        model_config=config.model, networks=networks
    ).to(device)

    # dataset
    def _create_iterator(dataset, for_train: bool):
        return MultiprocessIterator(
            dataset,
            config.train.batchsize,
            repeat=for_train,
            shuffle=for_train,
            n_processes=config.train.num_processes,
            dataset_timeout=60 * 15,
        )

    datasets = create_dataset(config.dataset)
    train_iter = _create_iterator(datasets["train"], for_train=True)
    test_iter = _create_iterator(datasets["test"], for_train=False)

    warnings.simplefilter("error", MultiprocessIterator.TimeoutWarning)

    # optimizer
    style_transfer_optimizer = create_optimizer(
        config=config.train.style_transfer_optimizer, model=networks.style_transfer
    )
    mapping_network_optimizer = create_optimizer(
        config=config.train.mapping_network_optimizer, model=networks.mapping_network
    )
    style_encoder_optimizer = create_optimizer(
        config=config.train.style_encoder_optimizer, model=networks.style_encoder
    )
    discriminator_optimizer = create_optimizer(
        config=config.train.discriminator_optimizer, model=networks.discriminator
    )

    # updater
    updater = Updater(
        iterator=train_iter,
        optimizer=dict(
            style_transfer=style_transfer_optimizer,
            mapping_network=mapping_network_optimizer,
            style_encoder=style_encoder_optimizer,
            discriminator=discriminator_optimizer,
        ),
        model=dict(
            generator=generator_model,
            discriminator=discriminator_model,
            moving_generator=moving_generator_model,
        ),
        moving_average_rate=config.train.moving_average_rate,
        device=device,
    )

    # trainer
    trigger_log = (config.train.log_iteration, "iteration")
    trigger_snapshot = (config.train.snapshot_iteration, "iteration")
    trigger_stop = (
        (config.train.stop_iteration, "iteration")
        if config.train.stop_iteration is not None
        else None
    )

    trainer = Trainer(updater, stop_trigger=trigger_stop, out=output)

    def eval_func(**kwargs):
        generator_model.forward_with_latent(**kwargs)
        generator_model.forward_with_reference(**kwargs)
        discriminator_model.forward_with_latent(**kwargs)
        discriminator_model.forward_with_reference(**kwargs)
        moving_generator_model.forward_with_latent(**kwargs)
        moving_generator_model.forward_with_reference(**kwargs)

    ext = extensions.Evaluator(
        test_iter,
        target=dict(
            generator=generator_model,
            discriminator=discriminator_model,
            moving_generator=moving_generator_model,
        ),
        eval_func=eval_func,
        device=device,
    )
    trainer.extend(ext, name="test", trigger=trigger_log)

    def add_snapshot_object(target, name):
        ext = extensions.snapshot_object(
            target, filename=name + "_{.updater.iteration}.pth"
        )
        trainer.extend(ext, trigger=trigger_snapshot)

    add_snapshot_object(networks.style_transfer, "style_transfer")
    add_snapshot_object(networks.mapping_network, "mapping_network")
    add_snapshot_object(networks.style_encoder, "style_encoder")

    trainer.extend(extensions.FailOnNonNumber(), trigger=trigger_log)
    trainer.extend(extensions.LogReport(trigger=trigger_log))
    trainer.extend(
        extensions.PrintReport(
            ["iteration", "generator/latent/loss", "test/generator/latent/loss"]
        ),
        trigger=trigger_log,
    )

    if config.train.model_config_linear_shift is not None:
        ext = ObjectLinearShift(
            **config.train.model_config_linear_shift, target=config.model
        )
        trainer.extend(
            ext, trigger=(1, "iteration"),
        )

    ext = TensorboardReport(writer=SummaryWriter(Path(output)))
    trainer.extend(ext, trigger=trigger_log)

    (output / "generator_struct.txt").write_text(repr(generator_model))
    (output / "discriminator_struct.txt").write_text(repr(discriminator_model))

    if trigger_stop is not None:
        trainer.extend(extensions.ProgressBar(trigger_stop))

    return trainer
