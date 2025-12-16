import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_pos_emb

class W2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projections for Mean Stream
        self.q_mu_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_mu_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_mu_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_mu_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Projections for Uncertainty (Log Sigma) Stream
        self.q_sigma_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_sigma_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_sigma_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_sigma_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Learnable Temperature
        self.tau = nn.Parameter(torch.tensor(1.0)) # Initial value, typically tuned
        self.epsilon = 1e-6

    def forward(
        self,
        hidden_states_mu,
        hidden_states_log_sigma,
        rotary_emb_outputs, # (cos, sin)
        attention_mask=None
    ):
        bsz, q_len, _ = hidden_states_mu.size()
        
        # 1. Projections
        # Mean Stream
        query_mu = self.q_mu_proj(hidden_states_mu)
        key_mu = self.k_mu_proj(hidden_states_mu)
        value_mu = self.v_mu_proj(hidden_states_mu)

        # Log Sigma Stream
        query_log_sigma = self.q_sigma_proj(hidden_states_log_sigma)
        key_log_sigma = self.k_sigma_proj(hidden_states_log_sigma)
        value_log_sigma = self.v_sigma_proj(hidden_states_log_sigma)

        # Reshape to [bs, num_heads, seq_len, head_dim]
        query_mu = query_mu.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_mu = key_mu.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_mu = value_mu.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_log_sigma = query_log_sigma.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_log_sigma = key_log_sigma.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_log_sigma = value_log_sigma.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE (Applied only to Mean Stream)
        cos, sin = rotary_emb_outputs
        query_mu, key_mu = apply_rotary_pos_emb(query_mu, key_mu, cos, sin)

        # 3. Wasserstein-2 Distance Calculation
        
        # 3.1 Mean Term: ||q||^2 + ||k||^2 - 2 <q, k>
        # Note: RoPE preserves norms, so ||q_rotated|| == ||q_original||. 
        # But we use the rotated ones for the dot product.
        
        # Dot product term: (bs, heads, q_len, head_dim) @ (bs, heads, head_dim, k_len) -> (bs, heads, q_len, k_len)
        dot_mu = torch.matmul(query_mu, key_mu.transpose(-1, -2))
        
        # Squared norms
        sq_q_mu = torch.sum(query_mu ** 2, dim=-1, keepdim=True) # (bs, heads, q_len, 1)
        sq_k_mu = torch.sum(key_mu ** 2, dim=-1, keepdim=True).transpose(-1, -2) # (bs, heads, 1, k_len)
        
        # Full Mean Distance Squared
        dist_mu_sq = sq_q_mu + sq_k_mu - 2 * dot_mu

        # 3.2 Sigma Term: ||sigma_q - sigma_k||^2
        # sigma = sqrt(exp(log_sigma)) = exp(0.5 * log_sigma)
        sigma_q = torch.exp(0.5 * query_log_sigma)
        sigma_k = torch.exp(0.5 * key_log_sigma)
        
        # Distance calculation via expansion similar to mean term for efficiency or direct? 
        # Expansion: ||sigma_q||^2 + ||sigma_k||^2 - 2 <sigma_q, sigma_k>
        # This allows matrix multiplication which is faster than broadcasting subtract + norm.
        
        dot_sigma = torch.matmul(sigma_q, sigma_k.transpose(-1, -2))
        sq_q_sigma = torch.sum(sigma_q ** 2, dim=-1, keepdim=True)
        sq_k_sigma = torch.sum(sigma_k ** 2, dim=-1, keepdim=True).transpose(-1, -2)
        
        dist_sigma_sq = sq_q_sigma + sq_k_sigma - 2 * dot_sigma

        # 4. Attention Score
        # S = - (D_mu^2 + D_sigma^2) / tau
        # Ensure tau is positive and non-zero
        denom = torch.abs(self.tau) + self.epsilon
        attn_scores = - (dist_mu_sq + dist_sigma_sq) / denom

        if attention_mask is not None:
             # attention_mask shape assumed: [bs, 1, 1, k_len] or [bs, 1, q_len, k_len]
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)

        # 5. Output Aggregation
        
        # 5.1 Mean Output: standard weighted sum
        # weights: (bs, heads, q_len, k_len)
        # value_mu: (bs, heads, k_len, head_dim)
        attn_output_mu = torch.matmul(attn_weights, value_mu)

        # 5.2 Uncertainty Output: Geometric average in original domain => Arithmetic average in log domain
        # o_l = sum(A * v_l)
        attn_output_log_sigma = torch.matmul(attn_weights, value_log_sigma)

        # Reshape and project out
        attn_output_mu = attn_output_mu.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output_log_sigma = attn_output_log_sigma.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)

        output_mu = self.o_mu_proj(attn_output_mu)
        output_log_sigma = self.o_sigma_proj(attn_output_log_sigma)

        return output_mu, output_log_sigma
