import sys
import os

# Add parent directory to path to allow imports from w2_rope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import json
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, asdict

# Import W2 components
from w2_rope.block import W2TransformerBlock
from w2_rope.rope import RotaryEmbedding, apply_rotary_pos_emb

# ==========================================
# 1. Config & Utils
# ==========================================

@dataclass
class ModelConfig:
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    num_hidden_layers: int
    rms_norm_eps: float = 1e-6

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. Standard Attention & Block (Baseline)
# ==========================================

class StandardAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
    def forward(self, hidden_states, rotary_emb_outputs, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = rotary_emb_outputs
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
             scores = scores + attention_mask
             
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(output)

class StandardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class StandardBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        from w2_rope.ffn import RMSNorm # Reuse same RMSNorm
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = StandardAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = StandardMLP(config)
        
    def forward(self, hidden_states, rotary_emb_outputs, attention_mask=None):
        normed = self.norm1(hidden_states)
        attn_out = self.attn(normed, rotary_emb_outputs, attention_mask)
        hidden_states = hidden_states + attn_out
        
        normed = self.norm2(hidden_states)
        ffn_out = self.mlp(normed)
        hidden_states = hidden_states + ffn_out
        
        return hidden_states

# ==========================================
# 3. Model Wrappers
# ==========================================

class MiniTransformer(nn.Module):
    def __init__(self, config, block_type="standard"):
        super().__init__()
        self.config = config
        self.block_type = block_type
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads)
        
        # W2 needs a second stream start
        if block_type == "w2":
            # W2++ uses standard single stream embedding and block structure
            # The Gaussian logic is encapsulated inside the Attention Head
            self.layers = nn.ModuleList([W2TransformerBlock(config) for _ in range(config.num_hidden_layers)])
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.layers = nn.ModuleList([StandardBlock(config) for _ in range(config.num_hidden_layers)])
            self.final_norm = nn.LayerNorm(config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            
    def forward(self, input_ids, return_embeddings=False):
        bsz, seq_len = input_ids.shape
        
        # RoPE
        rotary_out = self.rotary(torch.zeros(1, 1, 1, 1).to(input_ids.device), seq_len=seq_len) 
        
        # Causal Mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(input_ids.device)
        attn_mask = torch.zeros(seq_len, seq_len, device=input_ids.device).masked_fill(mask, float("-inf"))
        attn_mask = attn_mask.view(1, 1, seq_len, seq_len)
        
        h = self.embed(input_ids)
        
        for layer in self.layers:
            h = layer(h, rotary_out, attn_mask)
            
        h = self.final_norm(h)
        
        if return_embeddings:
            return h
            
        logits = self.lm_head(h)
        return logits

# ==========================================
# 4. Data Generator (Associative Recall)
# ==========================================

def get_batch(bsz, seq_len, vocab_size, device):
    half_len = seq_len // 2
    data = torch.randint(0, vocab_size, (bsz, half_len), device=device)
    data = torch.cat([data, data], dim=1) 
    x = data[:, :-1]
    y = data[:, 1:]
    return x, y

# ... (previous imports)

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.orthogonal_(module.weight)

def train_and_evaluate(
    config: ModelConfig,
    block_type: str,
    steps: int,
    batch_size: int = 32,
    seq_len: int = 128,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    learning_rate: float = 1e-3,
    use_orthogonal_init: bool = False
) -> Dict[str, Any]:
    
    print(f"Initializing {block_type} model...", flush=True)
    model = MiniTransformer(config, block_type).to(device)
    
    if use_orthogonal_init:
        print("Applying Orthogonal Initialization...", flush=True)
        model.apply(init_weights)
        
    n_params = count_parameters(model)
    print(f"Params: {n_params}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # ... (rest of function unchanged)
    
    start_step = 0
    metrics_log = []
    
    # Checkpoint Loading
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_step = ckpt['step']
        metrics_log = ckpt.get('metrics', [])
        print(f"Resuming from step {start_step}", flush=True)
    
    if start_step >= steps:
        print("Training already completed for this scheduled run.", flush=True)
        return {
            "params": n_params,
            "metrics": metrics_log,
            "final_loss": metrics_log[-1]['loss'] if metrics_log else None
        }

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    losses = []
    
    try:
        model.train()
        for i in range(start_step, steps):
            x, y = get_batch(batch_size, seq_len, config.vocab_size, device)
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (i + 1) % 100 == 0:
                avg_loss = sum(losses[-100:]) / 100
                elapsed = time.time() - start_time
                max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                print(f"Step {i+1}/{steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s | Mem: {max_mem:.2f}MB", flush=True)
                
                metric_entry = {
                    "step": i + 1,
                    "loss": avg_loss,
                    "time_elapsed": elapsed,
                    "vram_mb": max_mem
                }
                metrics_log.append(metric_entry)
                
                # Save Checkpoint
                if checkpoint_path:
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'step': i + 1,
                        'metrics': metrics_log,
                        'config': asdict(config)
                    }, checkpoint_path)
                    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "params": n_params,
        "metrics": metrics_log,
        "total_time": total_time,
        "final_loss": metrics_log[-1]['loss'] if metrics_log else None
    }

if __name__ == "__main__":
    # Quick test if run directly
    config = ModelConfig(64, 4, 256, 1024, 512, 2)
    train_and_evaluate(config, "w2", 100, checkpoint_path="test_ckpt.pt")
