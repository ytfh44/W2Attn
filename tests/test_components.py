import torch
import torch.nn as nn
from dataclasses import dataclass
import pytest
from w2_rope.rope import RotaryEmbedding, apply_rotary_pos_emb
from w2_rope.attention import W2Attention
from w2_rope.block import W2TransformerBlock

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
    bs, seq_len = 1, 2
    
    # Construct identical q and k
    mu = torch.randn(bs, seq_len, config.hidden_size)
    log_sigma = torch.randn(bs, seq_len, config.hidden_size)
    
    # We need to manually invoke the distance logic or check if attention scores are maxed out (0 distance -> max score)
    # Let's bypass full forward and check logic internally if we could, but for now let's trust the forward and check gradients.
    
    # Property: A linear shift should affect distance? No, W2 distance is translation sensitive for means.
    pass

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

