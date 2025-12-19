import torch
import torch.nn as nn
from .attention import W2Attention, StandardAttention
from .ffn import RMSNorm, FeedForward

class W2TransformerBlock(nn.Module):
    """
    W2 Block. Single Stream Input/Output.
    Attention uses Gaussian Kernel internally.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Standard Pre-Norm Architecture
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attention = W2Attention(config)
        
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, rotary_emb_outputs, attention_mask=None):
        # 1. Attention Block
        normed_hidden = self.input_layernorm(hidden_states)
        attn_out = self.attention(
            normed_hidden,
            rotary_emb_outputs,
            attention_mask
        )
        hidden_states = hidden_states + attn_out
        
        # 2. FFN Block
        normed_hidden = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(normed_hidden)
        hidden_states = hidden_states + ffn_out
        
        return hidden_states

class StandardBlock(nn.Module):
    """
    Standard Transformer Block.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.attention = StandardAttention(config)
        
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, rotary_emb_outputs, attention_mask=None):
        normed_hidden = self.input_layernorm(hidden_states)
        attn_out = self.attention(
            normed_hidden,
            rotary_emb_outputs,
            attention_mask
        )
        hidden_states = hidden_states + attn_out
        
        normed_hidden = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(normed_hidden)
        hidden_states = hidden_states + ffn_out
        
        return hidden_states
