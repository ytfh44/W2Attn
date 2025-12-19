import sys
import os
import torch
import torch.nn as nn
import time
import math
import argparse
import urllib.request
from typing import List

# Add parent directory to path to allow imports from w2attn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w2attn.config import ModelConfig
from benches.bench_fairness import MiniTransformer, count_parameters, train_and_evaluate

# ==========================================
# Data Loading Utils
# ==========================================

class SimpleTokenizer:
    def __init__(self, text: str):
        # Create a simple char-level or word-level tokenizer
        # For simplicity and small vocab, let's use character level for this benchmark
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)
        
    def decode(self, t: torch.Tensor) -> str:
        return ''.join([self.itos[i.item()] for i in t])

def get_wikitext_data(split='valid', cache_dir='data'):
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"wikitext-2-{split}.txt"
    filepath = os.path.join(cache_dir, filename)
    
    url_map = {
        'valid': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt',
        'train': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt',
        'test': 'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/test.txt'
    }
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url_map[split], filepath)
        except Exception as e:
            print(f"Failed to download: {e}")
            return None
            
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_tinystories_data(cache_dir='data'):
    # Without datasets lib, this is harder. For now, implemented as placeholder
    # or fallback to wikitext if requested.
    print("TinyStories download without 'datasets' lib not implemented yet. Using Wikitext valid as fallback.")
    return get_wikitext_data('valid', cache_dir)

class TextDataset:
    def __init__(self, data_tensor, seq_len):
        self.data = data_tensor
        self.seq_len = seq_len
        
    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.seq_len, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x.to(device), y.to(device)

# ==========================================
# Main Benchmark Logic
# ==========================================

def run_pe_benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")
    
    # 1. Load Data
    text = None
    if args.dataset == 'wikitext':
        text = get_wikitext_data('valid') # Use valid for smaller quick bench
    elif args.dataset == 'tinystories':
        text = get_tinystories_data()
    else:
        # Synthetic is handled inside train_and_evaluate if we passed it there,
        # but here we want to use the same loop. 
        # Actually bench_fairness.py generates data on fly.
        pass

    vocab_size = args.vocab_size
    tokenizer = None
    dataset = None
    
    if text:
        tokenizer = SimpleTokenizer(text)
        vocab_size = tokenizer.vocab_size
        print(f"Loaded text data. Vocab size: {vocab_size}")
        data_tensor = tokenizer.encode(text)
        dataset = TextDataset(data_tensor, args.seq_len)

    # 2. Config
    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        vocab_size=vocab_size,
        max_position_embeddings=args.seq_len * 2, # ample room
        num_hidden_layers=args.layers,
        position_embedding_type=args.pe_type
    )
    
    # 3. Model
    print(f"Initializing model with PE: {args.pe_type}")
    model = MiniTransformer(config, block_type="w2").to(device)
    print(f"Params: {count_parameters(model)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Loop
    model.train()
    losses = []
    start_time = time.time()
    
    steps = args.steps
    
    for i in range(steps):
        if dataset:
            x, y = dataset.get_batch(args.batch_size, device)
        else:
            # Synthetic associative recall reuse
            from benches.bench_fairness import get_batch
            x, y = get_batch(args.batch_size, args.seq_len, vocab_size, device)
            
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (i+1) % 10 == 0:
            print(f"Step {i+1}/{steps} | Loss: {loss.item():.4f}")
            
    total_time = time.time() - start_time
    avg_loss = sum(losses) / len(losses)
    print(f"Done. Avg Loss: {avg_loss:.4f}, Time: {total_time:.2f}s, Throughput: {steps*args.batch_size*args.seq_len / total_time:.0f} tok/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe_type", type=str, default="rope", choices=["rope", "alibi", "relative", "none"])
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "wikitext", "tinystories"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1024) # Only for synthetic
    
    args = parser.parse_args()
    run_pe_benchmark(args)
