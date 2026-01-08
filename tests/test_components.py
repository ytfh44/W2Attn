import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pytest
from w2attn.rope import RotaryEmbedding, apply_rotary_pos_emb
from w2attn.attention import W2Attention
from w2attn.block import W2TransformerBlock

@dataclass
class Config:
    hidden_size: int = 64
    num_attention_heads: int = 4
    intermediate_size: int = 256
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 128

@pytest.fixture
def config():
    return Config()

def test_rope_shapes(config):
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings)
    
    bs, seq_len = 2, 10
    # Dummy query: [bs, heads, seq_len, head_dim]
    q = torch.randn(bs, config.num_attention_heads, seq_len, head_dim)
    k = torch.randn(bs, config.num_attention_heads, seq_len, head_dim)
    
    cos, sin = rope(q, seq_len=seq_len)
    
    assert cos.shape == (1, 1, seq_len, head_dim)
    assert sin.shape == (1, 1, seq_len, head_dim)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape

def test_attention_forward(config):
    attn = W2Attention(config)
    bs, seq_len = 2, 10
    
    # Inputs - single stream (unified hidden states)
    hidden_states = torch.randn(bs, seq_len, config.hidden_size)
    
    # RoPE
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(hidden_states, seq_len=seq_len)
    
    # Forward - single stream output
    out = attn(hidden_states, (cos, sin))
    
    assert out.shape == (bs, seq_len, config.hidden_size)
    
def test_block_forward(config):
    block = W2TransformerBlock(config)
    bs, seq_len = 2, 16
    
    # Single stream input
    hidden_states = torch.randn(bs, seq_len, config.hidden_size)
    
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(hidden_states, seq_len=seq_len)
    
    # Single stream output
    out = block(hidden_states, (cos, sin))
    
    assert out.shape == (bs, seq_len, config.hidden_size)

def test_w2_distance_symmetry(config):
    # Sanity check: Distance between identical distributions should be 0 (or close to 0 due to numerics)
    attn = W2Attention(config)
    bs, seq_len = 1, 4

    # Force identical projections so q and k represent the same distributions
    attn.k_proj.weight.data.copy_(attn.q_proj.weight.data)

    hidden_states = torch.randn(bs, seq_len, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(hidden_states, seq_len=seq_len)

    query_states = attn.q_proj(hidden_states).view(bs, seq_len, attn.num_heads, 2 * attn.head_dim)
    key_states = attn.k_proj(hidden_states).view(bs, seq_len, attn.num_heads, 2 * attn.head_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    q_mu, q_sigma_raw = query_states.chunk(2, dim=-1)
    k_mu, k_sigma_raw = key_states.chunk(2, dim=-1)

    q_sigma = F.softplus(q_sigma_raw) + 1e-4
    k_sigma = F.softplus(k_sigma_raw) + 1e-4

    q_weighted = q_mu / torch.sqrt(q_sigma)
    k_weighted = k_mu / torch.sqrt(k_sigma)

    if attn.pe_type == "rope":
        q_weighted, k_weighted = apply_rotary_pos_emb(q_weighted, k_weighted, cos, sin)

    q_sq = (q_weighted ** 2).sum(dim=-1, keepdim=True)
    k_sq = (k_weighted ** 2).sum(dim=-1, keepdim=True)
    dot_term = torch.matmul(q_weighted, k_weighted.transpose(-2, -1))
    weighted_dist = q_sq + k_sq.transpose(-2, -1) - 2 * dot_term

    log_det_q = torch.log(q_sigma).sum(dim=-1, keepdim=True)
    log_det_k = torch.log(k_sigma).sum(dim=-1, keepdim=True)
    log_det = log_det_q + log_det_k.transpose(-2, -1)

    attn_scores = -0.5 * (weighted_dist + log_det)

    assert torch.allclose(attn_scores, attn_scores.transpose(-1, -2), atol=1e-5, rtol=1e-5)

def test_gradients(config):
    block = W2TransformerBlock(config)
    bs, seq_len = 2, 5
    hidden_states = torch.randn(bs, seq_len, config.hidden_size, requires_grad=True)
    
    head_dim = config.hidden_size // config.num_attention_heads
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(hidden_states, seq_len=seq_len)
    
    out = block(hidden_states, (cos, sin))
    
    loss = out.mean()
    loss.backward()
    
    assert hidden_states.grad is not None
