import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
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

def run_tests():
    print("Running Verification Tests...")
    config = Config()
    
    # 1. RoPE Test
    print("Test 1: RoPE Shapes...", end="")
    try:
        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(head_dim, max_position_embeddings=config.max_position_embeddings)
        bs, seq_len = 2, 10
        q = torch.randn(bs, config.num_attention_heads, seq_len, head_dim)
        k = torch.randn(bs, config.num_attention_heads, seq_len, head_dim)
        cos, sin = rope(q, seq_len=seq_len)
        
        assert cos.shape == (1, 1, seq_len, head_dim), f"Cos shape mismatch: {cos.shape}"
        assert sin.shape == (1, 1, seq_len, head_dim), f"Sin shape mismatch: {sin.shape}"
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 2. Attention Forward
    print("Test 2: Attention Forward...", end="")
    try:
        attn = W2Attention(config)
        bs, seq_len = 2, 10
        hidden_states = torch.randn(bs, seq_len, config.hidden_size)
        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(hidden_states, seq_len=seq_len)
        
        out_hidden = attn(hidden_states, (cos, sin))
        
        assert out_hidden.shape == (bs, seq_len, config.hidden_size)
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 3. Block Forward
    print("Test 3: Block Forward...", end="")
    try:
        block = W2TransformerBlock(config)
        bs, seq_len = 2, 16
        hidden_states = torch.randn(bs, seq_len, config.hidden_size)
        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(hidden_states, seq_len=seq_len)
        
        out_hidden = block(hidden_states, (cos, sin))
        
        assert out_hidden.shape == (bs, seq_len, config.hidden_size)
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")
        import traceback
        traceback.print_exc()

    # 4. Gradients
    print("Test 4: Gradients...", end="")
    try:
        block = W2TransformerBlock(config)
        bs, seq_len = 2, 5
        hidden_states = torch.randn(bs, seq_len, config.hidden_size, requires_grad=True)
        head_dim = config.hidden_size // config.num_attention_heads
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(hidden_states, seq_len=seq_len)
        
        out_hidden = block(hidden_states, (cos, sin))
        loss = out_hidden.mean()
        loss.backward()
        
        assert hidden_states.grad is not None
        print(" PASSED")
    except Exception as e:
        print(f" FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()
