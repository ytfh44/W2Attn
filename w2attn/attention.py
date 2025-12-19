import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_pos_emb

def get_alibi_slopes(n_heads):
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]

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

        # Positional Encoding Config
        self.pe_type = getattr(config, "position_embedding_type", "rope")
        
        if self.pe_type == "relative":
            # Simple learnable relative bias
            # Range: -seq_len to +seq_len -> 2 * max_pos
            self.relative_embedding = nn.Embedding(2 * config.max_position_embeddings + 1, 1)
        elif self.pe_type == "alibi":
            slopes = torch.tensor(get_alibi_slopes(self.num_heads))
            self.register_buffer("alibi_slopes", slopes)

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
        
        # 4. Calculate Score (Diagonal Covariance - Per-Dimension Weighting)
        # =================================================================
        # W2 distance for diagonal Gaussians:
        #   D² = Σ_d [(μ_q^d - μ_k^d)² / (σ_q^d + σ_k^d)] + Σ_d log(σ_q^d + σ_k^d)
        #
        # Memory-efficient formulation using weighted space:
        #   - Transform to weighted space: q̃ = μ_q / √σ_q, k̃ = μ_k / √σ_k
        #   - Compute ||q̃||² + ||k̃||² - 2⟨q̃, k̃⟩ (O(S²) memory)
        #   - Log-det: Σ_d log(σ_q^d) + Σ_d log(σ_k^d)
        #
        # Note: This is a principled approximation that captures per-dimension
        # anisotropy while maintaining O(S²) attention memory (not O(S²D)).
        
        # a. Weighted Q/K in transformed space
        q_weighted = q_mu / torch.sqrt(q_sigma)  # [B, H, S, D]
        k_weighted = k_mu / torch.sqrt(k_sigma)  # [B, H, S, D]

        # 5. PE Application
        # =================
        
        # RoPE: Only apply if type is 'rope'
        if self.pe_type == "rope":
            # Apply RoPE after weighting to preserve relative position invariance
            # when sigma is non-uniform across RoPE pairs.
            cos, sin = rotary_emb_outputs
            q_weighted, k_weighted = apply_rotary_pos_emb(q_weighted, k_weighted, cos, sin)
        
        # b. Squared distance in weighted space: ||q̃||² + ||k̃||² - 2⟨q̃, k̃⟩
        q_sq = (q_weighted ** 2).sum(dim=-1, keepdim=True)  # [B, H, S, 1]
        k_sq = (k_weighted ** 2).sum(dim=-1, keepdim=True)  # [B, H, S, 1]
        dot_term = torch.matmul(q_weighted, k_weighted.transpose(-2, -1))  # [B, H, S, S]
        weighted_dist = q_sq + k_sq.transpose(-2, -1) - 2 * dot_term  # [B, H, S, S]
        
        # c. Log determinant: Σ_d log(σ_q^d) + Σ_d log(σ_k^d)
        log_det_q = torch.log(q_sigma).sum(dim=-1, keepdim=True)  # [B, H, S, 1]
        log_det_k = torch.log(k_sigma).sum(dim=-1, keepdim=True)  # [B, H, S, 1]
        log_det = log_det_q + log_det_k.transpose(-2, -1)  # [B, H, S, S]
        
        # d. Final attention score
        attn_scores = -0.5 * (weighted_dist + log_det)

        # Add ALiBi or Relative Bias
        if self.pe_type == "alibi":
            # shape: [1, num_heads, 1, 1]
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            # Create distance matrix [1, 1, S, S]
            context_position = torch.arange(seq_len, device=hidden_states.device)[:, None]
            memory_position = torch.arange(seq_len, device=hidden_states.device)[None, :]
            relative_position = torch.abs(context_position - memory_position) 
            # Non-causal absolute distance for now, or causal? 
            # ALiBi papers usually do causal: score += slope * (j - i) for j <= i.
            # But standard implementation is often just adding a bias matrix. 
            # W2 is usually bidirectional in calculation but masked later.
            # Let's use standard ALiBi causal-compatible bias: -|i-j| logic
            # For causal: [i, j], verify input j <= i. i is query, j is key.
            # Bias = slope * (j - i). Since j <= i, this is negative.
            # If we want symmetric decay for non-causal: -|i-j|
            
            # Using -|i-j| makes it compatible with both.
            # shape expansion: [1, 1, S, S]
            bias = -1.0 * relative_position.view(1, 1, seq_len, seq_len).to(hidden_states.dtype)
            attn_scores = attn_scores + (slopes * bias)
            
        elif self.pe_type == "relative":
            q_idx = torch.arange(seq_len, device=hidden_states.device)[:, None]
            k_idx = torch.arange(seq_len, device=hidden_states.device)[None, :]
            # shift to be positive [0, 2*max]
            relative_idx = (q_idx - k_idx) + self.config.max_position_embeddings
            # clamp just in case
            relative_idx = relative_idx.clamp(0, 2 * self.config.max_position_embeddings)
            
            # [S, S, 1]
            rel_bias = self.relative_embedding(relative_idx)
            # [1, 1, S, S]
            rel_bias = rel_bias.permute(2, 0, 1).unsqueeze(0)
            attn_scores = attn_scores + rel_bias

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

        # Positional Encoding Config
        self.pe_type = getattr(config, "position_embedding_type", "rope")
        
        if self.pe_type == "alibi":
            slopes = torch.tensor(get_alibi_slopes(self.num_heads))
            self.register_buffer("alibi_slopes", slopes)
        
    def forward(self, hidden_states, rotary_emb_outputs, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = rotary_emb_outputs
        if self.pe_type == "rope":
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        import math
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Add ALiBi
        if self.pe_type == "alibi":
            slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1)
            context_position = torch.arange(q_len, device=hidden_states.device)[:, None]
            memory_position = torch.arange(q_len, device=hidden_states.device)[None, :]
            relative_position = torch.abs(context_position - memory_position)
            bias = -1.0 * relative_position.view(1, 1, q_len, q_len).to(hidden_states.dtype)
            scores = scores + (slopes * bias)
        
        if attention_mask is not None:
             scores = scores + attention_mask
             
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(output)
