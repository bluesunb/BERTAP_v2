from dataclasses import dataclass
from pathlib import Path
from src import BASE_DIR
from src.common.config_base import ConfigBase
from typing import Literal


@dataclass
class ModelConfig(ConfigBase):
    # Environment
    # transition_dim: int     # Dimension of the input of the model (= obs_dim + act_dim + rew + done)
    transition_dim: int     # Dimension of the transition MDP set (= obs_dim + act_dim + done)
    observation_dim: int    # Dimension of the observation space
    action_dim: int         # Dimension of the action space
    goal_dim: int = 0       # Dimension of the goal space, 0 for no goal
    hierarchical_goal: bool = False  # Whether to use a hierarchical goal space

    # Attentions
    causal: bool = False        # Whether to use a causal attention mask
    emb_dim: int = 1024         # Dimension of the embeddings in the model
    n_heads: int = 16           # Number of attention heads
    n_layers: int = 12          # Number of layers of the Transformer
    ff_dim: int = 4096          # Dimension of the feed-forward layers
    attn_pdrop: float = 0.1     # Dropout rate for attention weights
    resid_pdrop: float = 0.1    # Dropout rate for all residual connections
    emb_pdrop: float = 0.1      # Dropout rate for the embeddings layers
    ff_pdrop: float = 0.1       # Dropout rate for the feed-forward layers

    # BERT
    n_special_tokens: int = 4   # Number of special tokens for the BERT model
    _pad_token: int = 0          # Token for padding
    _cls_token: int = 1          # Token for the classification
    _mask_token: int = 2         # Token for masking
    _sep_token: int = 3          # Token for separating sequences
    shift: int = 0              # Shift for the trajectory tokens, token=`shift` is token=0
    modify_prob: float = 0.15   # Probability of modifying the input sequence
    mask_prob: float = 0.7      # Probability of masking a token
    replace_prob: float = 0.1   # Probability of replacing a token
    delete_prob: float = 0.1    # Probability of deleting a token

    # Vector Quantization
    traj_emb_dim: int = 512     # Dimension of the trajectory embeddings = VQ embedding size
    n_traj_tokens: int = 360    # Number of tokens for the trajectory codebook
    ma_update: bool = True      # Whether to update the moving average of the codebook

    # VAE
    seq_len: int = 24           # Length of the subsequence for the transformer
    latent_step: int = 3        # Number of steps to aggregate the latent code
    bottleneck: str = 'conv'    # Type of bottleneck to use for trajectory representation
    goal_conditional: bool = True   # Whether to use a conditional model
    multi_modal: bool = False   # Whether to use separated modalities for the input
    enc_gc: bool = True         # Whether to use the goal-conditioned encoder
    dec_gc: bool = True         # Whether to use the goal-conditioned decoder

    # Masking
    n_transition_mask: int = 0  # Number of masks to apply to the input transitions
    n_latent_mask: int = 1      # Number of masks to apply to the latent code
    min_transition_mask: int = 0    # Minimum number of masks to apply to the input transitions
    min_latent_mask: int = 1    # Minimum number of masks to apply to the latent code
    mask_schedule: str = 'cosine'   # Schedule for the mask annealing

    # Sampling
    sample_temperature: float = 4.5     # Temperature for ids sampling in the transformer

    # Losses
    pos_weight: float = 1.0             # Loss weight for the position modalities
    action_weight: float = 5.0          # Loss weight for the action modalities
    masked_pred_weight: float = 1.0     # Loss weight for the masked prediction
    commit_weight: float = 0.1          # Loss weight for the commitment loss
    vq_weight: float = 1.0              # Loss weight for the VQ loss

    # (Optional) Goal VQ Encoding
    goal_emb_dim: int = 256     # Dimension of the goal embeddings
    n_goal_tokens: int = 128    # Number of tokens in the goal codebook

    # (Special) Pretrained VAE
    vae_path: str = None        # Path to a pretrained VAE model

    def __post_init__(self):
        # super().__post_init__()
        assert self.transition_dim >= self.observation_dim + self.action_dim, \
            (f'Transition dim({self.transition_dim}) must be greater than '
             f'observation({self.observation_dim}) and action({self.action_dim}) dim')
        
    @property
    def reduced_len(self) -> int:
        return self.seq_len // self.latent_step

    @property
    def goal_conditioned(self) -> bool:
        return self.enc_gc or self.dec_gc
    
    # @property
    # def goal_input_dim(self) -> int:
    #     """Dimension of the goal as model input"""
    #     return self.goal_dim * (1 + self.hierarchical_goal)
    
    @property
    def vocab_size(self) -> int:
        return self.n_traj_tokens + self.n_special_tokens
    
    @property
    def pad_token(self) -> int:
        return self._pad_token + self.n_traj_tokens
    
    @property
    def mask_token(self) -> int:
        return self._mask_token + self.n_traj_tokens
    
    @property
    def cls_token(self) -> int:
        return self._cls_token + self.n_traj_tokens
    
    @property
    def sep_token(self) -> int:
        return self._sep_token + self.n_traj_tokens

@dataclass
class DatasetConfig(ConfigBase):
    env_name: str
    seq_len: int = 24           # Length of the sampled subsequence from dataloader
    disable_goal: bool = False  # Whether to disable the goal space
    pad_value: float = 0.0      # Value to pad the sequences
    min_valid_len: int = 3        # Minimum length of non-padded sequences
    terminal_key: str = 'dones_float'   # Key for the terminal signal in the dataset
    terminal: bool = True       # Whether to use the terminal signal in the dataset
    p_true_goal: float = 1.0    # Probability of using the true goal in the dataset
    p_sub_goal: float = 0.0     # Probability of using a sub-goal in the dataset
    goal_conditioned: bool = True   # Whether to use a goal-conditioned model
    hierarchical_goal: bool = False  # Whether to use a hierarchical goal space


@dataclass
class TrainConfig(ConfigBase):
    exp_name: str = 'no_name'
    seed: int = 0
    batch_size: int = 32
    n_epochs: int = 10
    learning_rate: float = 1e-4
    grad_norm_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.0
    scheduler_name: Literal["constant", "onecycle", "bertwarmup"] = "constant"
    scheduler_params: dict = None

    @property
    def save_dir(self) -> Path:
        return Path(BASE_DIR["save"]) / self.exp_name

    @property
    def project_root(self) -> Path:
        return Path(BASE_DIR["project"])


class TotalConfigs:
    data_config: DatasetConfig
    model_config: ModelConfig
    train_config: TrainConfig

    def __init__(self,
                 data_config: DatasetConfig,
                 model_config: ModelConfig,
                 train_config: TrainConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config

    def get_dict(self):
        return {'data_config': self.data_config.get_dict(),
                'model_config': self.model_config.get_dict(),
                'train_config': self.train_config.get_dict()}
    
    def save(self, save_path: Path):
        save_path.mkdir(parents=True, exist_ok=True)
        self.data_config.save(save_path / 'data_config.pkl')
        self.model_config.save(save_path / 'model_config.pkl')
        self.train_config.save(save_path / 'train_config.pkl')
        print(f'Saved configs to {save_path}')

    @classmethod
    def load(cls, path: Path) -> 'TotalConfigs':
        data_config = DatasetConfig.load(path / 'data_config.pkl')
        model_config = ModelConfig.load(path / 'model_config.pkl')
        train_config = TrainConfig.load(path / 'train_config.pkl')
        config = cls(data_config, model_config, train_config)
        return config
