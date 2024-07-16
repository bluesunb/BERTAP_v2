from dataclasses import dataclass
from src import BASE_DIR
from src.common.config_base import ConfigBase


@dataclass
class ModelConfig(ConfigBase):
    observation_dim: int
    action_dim: int
    goal_dim: int
    window_size: int
    future_size: int
    state_action: bool = False
    
    causal: bool = False
    emb_dim: int = 256
    n_heads: int = 4
    ff_dim: int = 256 * 4
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    emb_pdrop: float = 0.1
    ff_pdrop: float = 0.1
    
    n_slots: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 1
    use_goal: bool = True
    
    mask_prob: float = 0.6
    
    @property
    def seq_len(self) -> int:
        return self.window_size + self.future_size
    
    @property
    def features_dim(self) -> int:
        return self.observation_dim + (self.action_dim if self.state_action else 0)