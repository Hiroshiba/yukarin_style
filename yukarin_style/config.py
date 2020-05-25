from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from yukarin_style.utility import dataclass_utility
from yukarin_style.utility.git_utility import get_commit_id, get_branch_name


@dataclass
class DatasetConfig:
    sampling_length: int
    spectrogram_glob: str
    silence_glob: str
    min_not_silence_length: int
    seed: int
    num_train: Optional[int]
    num_test: int
    evaluate_times: Optional[int]


@dataclass
class StyleTransferConfig:
    hidden_size: int
    kernel_size: int
    residual_block_num: int
    adaptive_residual_block_num: int


@dataclass
class MappingNetworkConfig:
    hidden_size: int
    layer_num: int


@dataclass
class StyleEncoderConfig:
    min_hidden_size: int
    max_hidden_size: int
    kernel_size: int
    residual_block_num: int
    last_kernel_size: int


@dataclass
class DiscriminatorConfig:
    min_hidden_size: int
    max_hidden_size: int
    kernel_size: int
    residual_block_num: int
    last_kernel_size: int


@dataclass
class NetworkConfig:
    feature_size: int
    style_size: int
    latent_size: int
    style_transfer: StyleTransferConfig
    mapping_network: MappingNetworkConfig
    style_encoder: StyleEncoderConfig
    discriminator: DiscriminatorConfig


@dataclass
class ModelConfig:
    style_reconstruction_weight: float
    diversity_sensitive_weight: float
    cycle_consistency_weight: float
    identification_weight: float
    r1_weight: float


@dataclass
class TrainConfig:
    batchsize: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    style_transfer_optimizer: Dict[str, Any]
    mapping_network_optimizer: Dict[str, Any]
    style_encoder_optimizer: Dict[str, Any]
    discriminator_optimizer: Dict[str, Any]
    moving_average_rate: float
    model_config_linear_shift: Optional[Dict[str, Any]]
    num_processes: Optional[int] = None


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
