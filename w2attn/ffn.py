import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    """
    Standard SwiGLU FFN.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
