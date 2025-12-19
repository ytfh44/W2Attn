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
from benches.bench_fairness import MiniTransformer, count_parameters, train_and_evaluate, get_batch

# Reuse tokenizer and downloader from bench_pe.py
# Assuming they are robust enough now, duplicating for self-containment 
# or import if preferred. Let's just quick-import or copy small utils.

class SimpleTokenizer:
    def __init__(self, text: str):
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

class TextDataset:
    def __init__(self, data_tensor, seq_len):
        self.data = data_tensor
        self.seq_len = seq_len
        
    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.seq_len, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x.to(device), y.to(device)

def run_comparison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comparison on {device} | PE: {args.pe_type}")
    
    # Data Prep
    text = get_wikitext_data('valid')
    dataset = None
    if text:
        tokenizer = SimpleTokenizer(text)
        vocab_size = tokenizer.vocab_size
        print(f"Wikitext loaded. Vocab: {vocab_size}")
        dataset = TextDataset(tokenizer.encode(text), args.seq_len)
    else:
        print("Using synthetic data.")
        vocab_size = args.vocab_size

    # Config shared
    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        vocab_size=vocab_size,
        max_position_embeddings=args.seq_len * 2,
        num_hidden_layers=args.layers,
        position_embedding_type=args.pe_type
    )

    models_to_test = ["w2", "standard"]
    results = {}

    for block_type in models_to_test:
        print(f"\nEvaluating {block_type.upper()} + {args.pe_type.upper()}...")
        model = MiniTransformer(config, block_type=block_type).to(device)
        print(f"Params: {count_parameters(model)}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        start_time = time.time()
        losses = []
        
        for i in range(args.steps):
            if dataset:
                x, y = dataset.get_batch(args.batch_size, device)
            else:
                x, y = get_batch(args.batch_size, args.seq_len, vocab_size, device)
                
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        avg_loss = sum(losses) / len(losses)
        ellapsed = time.time() - start_time
        thp = (args.steps * args.batch_size * args.seq_len) / ellapsed
        
        print(f"Result {block_type}: Loss={avg_loss:.4f}, Thp={thp:.0f} tok/s")
        results[block_type] = {"loss": avg_loss, "thp": thp}

    print("\n=== FINAL COMPARISON ===")
    print(f"{'Model':<10} | {'Loss':<10} | {'Tok/s':<10}")
    print("-" * 35)
    for m, res in results.items():
        print(f"{m:<10} | {res['loss']:.4f}     | {res['thp']:.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe_type", type=str, default="alibi")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=1024)
    
    args = parser.parse_args()
    run_comparison(args)
