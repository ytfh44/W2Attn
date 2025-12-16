import torch
import torch.nn as nn
from .attention import W2Attention
from .ffn import W2FeedForward, RMSNorm

class W2TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Norms before Attention
        self.input_layernorm_mu = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_sigma = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        # Attention
        self.attention = W2Attention(config)
        
        # Norms before FFN
        self.post_attention_layernorm_mu = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_sigma = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        
        # FFN
        self.mlp = W2FeedForward(config)

    def forward(
        self,
        hidden_states_mu,
        hidden_states_log_sigma,
        rotary_emb_outputs,
        attention_mask=None
    ):
        # 1. Attention Block (Pre-Norm)
        normed_mu = self.input_layernorm_mu(hidden_states_mu)
        normed_sigma = self.input_layernorm_sigma(hidden_states_log_sigma)
        
        attn_out_mu, attn_out_sigma = self.attention(
            normed_mu,
            normed_sigma,
            rotary_emb_outputs,
            attention_mask
        )
        
        # Residual Connection
        hidden_states_mu = hidden_states_mu + attn_out_mu
        hidden_states_log_sigma = hidden_states_log_sigma + attn_out_sigma

        # 2. FFN Block (Pre-Norm)
        normed_mu_ffn = self.post_attention_layernorm_mu(hidden_states_mu)
        normed_sigma_ffn = self.post_attention_layernorm_sigma(hidden_states_log_sigma)
        
        ffn_out_mu, ffn_out_sigma = self.mlp(normed_mu_ffn, normed_sigma_ffn)
        
        # Residual Connection
        hidden_states_mu = hidden_states_mu + ffn_out_mu
        hidden_states_log_sigma = hidden_states_log_sigma + ffn_out_sigma
        
        return hidden_states_mu, hidden_states_log_sigma
