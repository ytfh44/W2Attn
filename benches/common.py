import torch
import torch.nn as nn
import time
import os
from dataclasses import asdict
from typing import Optional, Dict, Any, List

from w2_rope.model import LanguageModel
from w2_rope.config import ModelConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_associative_batch(bsz, seq_len, vocab_size, device):
    half_len = seq_len // 2
    data = torch.randint(0, vocab_size, (bsz, half_len), device=device)
    data = torch.cat([data, data], dim=1) 
    x = data[:, :-1]
    y = data[:, 1:]
    return x, y

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
    model = LanguageModel(config, block_type).to(device)
    
    if use_orthogonal_init:
        print("Applying Orthogonal Initialization...", flush=True)
        model.apply(init_weights)
        
    n_params = count_parameters(model)
    print(f"Params: {n_params}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
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
            x, y = get_associative_batch(batch_size, seq_len, config.vocab_size, device)
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
