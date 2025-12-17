import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_pos_emb

class W2Attention(nn.Module):
    """
    W2 Attention: Unified Stream with Gaussian Overlap Kernel.
    
    Logic:
    - Input: Hidden states H
    - Projection: H -> Q_mu, Q_sigma, K_mu, K_sigma, V
    - Score: - ||mu_q - mu_k||^2 / (sigma_q + sigma_k) - log(sigma_q + sigma_k)
    - Output: Standard weighted sum of V
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Unified projection: 
        # Output dim = num_heads * (head_dim + head_dim) -> Mu + Sigma
        # We project Mu and Sigma together for efficiency
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Initialize gradients for sigma part to be small for stability
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02)

    def forward(
        self,
        hidden_states,
        rotary_emb_outputs,
        attention_mask=None,
    ):
        bsz, seq_len, _ = hidden_states.size()

        # 1. Projections
        # Q, K: [Batch, Seq, Heads, HeadDim * 2]
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, 2 * self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, 2 * self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        query_states = query_states.transpose(1, 2) # [B, H, S, D*2]
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # 2. Split Mu and Sigma
        # mu: [..., :head_dim], sigma_raw: [..., head_dim:]
        q_mu, q_sigma_raw = query_states.chunk(2, dim=-1)
        k_mu, k_sigma_raw = key_states.chunk(2, dim=-1)
        
        # 3. Activation for Sigma
        # Softplus is safer than Exp. shape: [B, H, S, D]
        # Adding epsilon to prevent division by zero
        q_sigma = F.softplus(q_sigma_raw) + 1e-4
        k_sigma = F.softplus(k_sigma_raw) + 1e-4
        
        # 4. RoPE (Only apply to Mu)
        cos, sin = rotary_emb_outputs
        q_mu, k_mu = apply_rotary_pos_emb(q_mu, k_mu, cos, sin)
        
        # 5. Calculate Score (Gaussian Overlap / Variance Scaled Distance)
        # OPTIMIZED IMPLEMENTATION
        # We approximate sigma as a scalar (mean over dim) to avoid O(S^2 D) memory
        # Scores = -0.5 * ( ||mu_q - mu_k||^2 / (sigma_q + sigma_k) + D * log(sigma_q + sigma_k) )
        
        # a. Squared Euclidean Distance ||mu_q - mu_k||^2 = ||q||^2 + ||k||^2 - 2 <q, k>
        # q_mu: [B, H, S, D]
        q_sq = (q_mu ** 2).sum(dim=-1, keepdim=True) # [B, H, S, 1]
        k_sq = (k_mu ** 2).sum(dim=-1, keepdim=True) # [B, H, S, 1]
        
        # Dot term: [B, H, S, S]
        dot_term = torch.matmul(q_mu, k_mu.transpose(-2, -1))
        
        # Dist Sq: [B, H, S, S]
        # [S, 1] + [1, S] - [S, S]
        dist_sq = q_sq + k_sq.transpose(-2, -1) - 2 * dot_term
        
        # b. Sigma terms (Scalar Approximation)
        # q_sigma: [B, H, S, D] -> [B, H, S, 1]
        q_sigma_scalar = q_sigma.mean(dim=-1, keepdim=True)
        k_sigma_scalar = k_sigma.mean(dim=-1, keepdim=True)
        
        # Sigma Sum: [B, H, S, S]
        sigma_sum = q_sigma_scalar + k_sigma_scalar.transpose(-2, -1)
        
        # c. Combine
        # Weighted Distance: dist_sq / sigma_sum
        weighted_dist = dist_sq / sigma_sum
        
        # Log Det: D * log(sigma_sum)
        # Note: In standard W2, we sum log(sigma_i + sigma_j) over D dimensions.
        # Here we assume isotropic, so it's D * log(scalar_sigma_sum).
        # However, to keep scale comparable to naive (which sums D logs), we multiply by D.
        # But wait, naive was log(sigma_sum_vector).sum(-1). 
        # approximate: sum(log(scalar)) = D * log(scalar)
        log_det = self.head_dim * torch.log(sigma_sum)
        
        # Final Score
        attn_scores = -0.5 * (weighted_dist + log_det)

        if attention_mask is not None:
             # attention_mask shape: [bs, 1, 1, k_len] or similar
            attn_scores = attn_scores + attention_mask

        # 6. Softmax & Output
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)

        return attn_output

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
        
        import math
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
             scores = scores + attention_mask
             
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(output)
