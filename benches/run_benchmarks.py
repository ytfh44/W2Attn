import argparse
import os
import sys

# Add parent directory to path to allow imports from w2attn and benches
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn as nn
import time
from w2attn.config import ModelConfig
from w2attn.model import LanguageModel
from benches.common import train_and_evaluate, count_parameters
from benches.hierarchy import HierarchyGenerator

def run_fairness():
    print("Running Fairness Benchmark...", flush=True)
    # Default configs from bench_fairness.py (if simplified)
    # or just run a standard vs w2 comparison
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Example Config
    config = ModelConfig(
        hidden_size=64, num_attention_heads=4, intermediate_size=256,
        vocab_size=1024, max_position_embeddings=512, num_hidden_layers=2
    )
    
    print("Testing W2...", flush=True)
    train_and_evaluate(config, "w2", steps=500, device=device)
    
    print("Testing Standard...", flush=True)
    train_and_evaluate(config, "standard", steps=500, device=device)

def train_entailment(
    model_type: str,
    config: ModelConfig,
    hier_gen: HierarchyGenerator,
    steps: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda"
):
    print(f"\nTraining {model_type} on Entailment (LR={lr})...")
    model = LanguageModel(config, model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    metrics = []
    
    model.train()
    for step in range(steps):
        x, y = hier_gen.get_batch(32, device) 
        
        logits = model(x) # [B, 2, Vocab]
        last_token_logits = logits[:, -1, :] # [B, Vocab]
        relevant_logits = last_token_logits[:, 0:2] # [B, 2]
        
        loss = criterion(relevant_logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % 100 == 0:
            preds = torch.argmax(relevant_logits, dim=1)
            acc = (preds == y).float().mean().item()
            elapsed = time.time() - start_time
            print(f"Step {step+1}: Loss {loss.item():.4f}, Acc {acc:.2f}, Time {elapsed:.1f}s")
            metrics.append({"step": step+1, "loss": loss.item(), "acc": acc, "time": elapsed})
            
    return {"metrics": metrics, "final_acc": metrics[-1]["acc"] if metrics else 0.0, "final_loss": metrics[-1]["loss"] if metrics else 0.0}

def run_entailment():
    print("Running Entailment Benchmark...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = HierarchyGenerator(vocab_size=1024, max_depth=6)
    
    results = {}
    
    # 1. Standard Attention (Baseline)
    std_config = ModelConfig(
        hidden_size=96, num_attention_heads=6, intermediate_size=int(96*2.68),
        vocab_size=1024, max_position_embeddings=128, num_hidden_layers=2
    )
    res_std = train_entailment("standard", std_config, gen, steps=2000, lr=1e-3, device=device)
    results["std"] = res_std
    
    # 2. W2 Attention (Target)
    w2_config = ModelConfig(
        hidden_size=88, num_attention_heads=4, intermediate_size=int(88*2.68),
        vocab_size=1024, max_position_embeddings=128, num_hidden_layers=2
    )
    res_w2 = train_entailment("w2", w2_config, gen, steps=2000, lr=5e-4, device=device)
    results["w2"] = res_w2
    
    print("\n=== Final Report ===")
    print(f"Standard: Acc {res_std['final_acc']:.2f}, Loss {res_std['final_loss']:.4f}")
    print(f"W2      : Acc {res_w2['final_acc']:.2f}, Loss {res_w2['final_loss']:.4f}")

def run_micro():
    print("Running Micro Suite...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = [
        {"name": "base_seq128", "seq_len": 128, "w2_h": 64, "std_h": 96, "heads_w2": 4, "heads_std": 6, "steps": 500, "batch_size": 32},
        {"name": "long_seq512", "seq_len": 512, "w2_h": 64, "std_h": 96, "heads_w2": 4, "heads_std": 6, "steps": 500, "batch_size": 4},
        {"name": "large_h128",  "seq_len": 128, "w2_h": 128, "std_h": 192, "heads_w2": 8, "heads_std": 12, "steps": 500, "batch_size": 16},
        {"name": "deep_l4",     "seq_len": 128, "w2_h": 64, "std_h": 96, "heads_w2": 4, "heads_std": 6, "steps": 500, "layers": 4, "batch_size": 32},
    ]
    
    for exp in experiments:
        print(f"Exp: {exp['name']}")
        w2_config = ModelConfig(
            hidden_size=exp["w2_h"], num_attention_heads=exp["heads_w2"], intermediate_size=exp["w2_h"] * 4,
            vocab_size=1024, max_position_embeddings=2048, num_hidden_layers=exp.get("layers", 2)
        )
        std_config = ModelConfig(
            hidden_size=exp["std_h"], num_attention_heads=exp["heads_std"], intermediate_size=exp["std_h"] * 4,
            vocab_size=1024, max_position_embeddings=2048, num_hidden_layers=exp.get("layers", 2)
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        bsz = exp.get("batch_size", 32)
        train_and_evaluate(w2_config, "w2", steps=exp["steps"], seq_len=exp["seq_len"], batch_size=bsz, device=device)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        train_and_evaluate(std_config, "standard", steps=exp["steps"], seq_len=exp["seq_len"], batch_size=bsz, device=device)

def run_sweep():
    print("Running LR Sweep...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lrs = [1e-3, 5e-3]
    config = ModelConfig(
        hidden_size=64, num_attention_heads=4, intermediate_size=256,
        vocab_size=1024, max_position_embeddings=2048, num_hidden_layers=2
    )
    
    for lr in lrs:
        print(f"LR: {lr}")
        train_and_evaluate(config, "w2", steps=200, learning_rate=lr, device=device)

def run_optimized():
    print("Running Optimized Benchmark...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    std_config = ModelConfig(
        hidden_size=96, num_attention_heads=6, intermediate_size=96*4,
        vocab_size=1024, max_position_embeddings=2048, num_hidden_layers=2
    )
    train_and_evaluate(std_config, "standard", steps=500, learning_rate=1e-3, use_orthogonal_init=True, device=device)
    
    w2_config = ModelConfig(
        hidden_size=64, num_attention_heads=4, intermediate_size=64*4,
        vocab_size=1024, max_position_embeddings=2048, num_hidden_layers=2
    )
    train_and_evaluate(w2_config, "w2", steps=500, learning_rate=5e-3, use_orthogonal_init=True, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fairness", "entailment", "micro", "sweep", "optimized"], help="Benchmark mode")
    args = parser.parse_args()
    
    if args.mode == "fairness":
        run_fairness()
    elif args.mode == "entailment":
        run_entailment()
    elif args.mode == "micro":
        run_micro()
    elif args.mode == "sweep":
        run_sweep()
    elif args.mode == "optimized":
        run_optimized()
