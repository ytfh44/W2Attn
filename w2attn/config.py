from dataclasses import dataclass

@dataclass
class ModelConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    num_hidden_layers: int
    position_embedding_type: str = "rope" # "rope", "alibi", "relative", "none"
    rms_norm_eps: float = 1e-6
