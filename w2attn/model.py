import torch
import torch.nn as nn
from .block import W2TransformerBlock, StandardBlock
from .rope import RotaryEmbedding

class LanguageModel(nn.Module):
    def __init__(self, config, block_type="standard"):
        super().__init__()
        self.config = config
        self.block_type = block_type
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config.hidden_size // config.num_attention_heads)
        
        if block_type == "w2":
            block_class = W2TransformerBlock
        else:
            block_class = StandardBlock
            
        self.layers = nn.ModuleList([block_class(config) for _ in range(config.num_hidden_layers)])
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
