# W2Attn

A PyTorch implementation of **Wasserstein-2 Attention**.
This model replaces the traditional Dot-Product Attention in Transformers with a metric based on the **Wasserstein-2 distance** between Gaussian distributions.

## 📖 Core Principle

In W2 Attention, each token is modeled as a diagonal Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$. To maintain compatibility with standard Transformer architectures and save memory, we use a **Unified Residual Stream** design:

1.  **Unified Stream**: The network passes a single hidden state vector $\mathbf{h} \in \mathbb{R}^d$ between layers.
2.  **Internal Projection**: Inside the W2 Attention layer, the hidden state is projected and split into **mean ($\boldsymbol{\mu}$)** and **uncertainty ($\boldsymbol{\sigma}$)** components.
3.  **W2 Attention Calculation**:
    Attention scores are determined by the negative $W_2^2$ distance between distributions:
    $$
    S_{m,n} = - \frac{D_{\mu}^2(m, n) + D_{\sigma}^2}{\tau}
    $$
    where:
    -   $D_{\mu}^2$: Euclidean distance between mean vectors (incorporating positional information).
    -   $D_{\sigma}^2$: Euclidean distance between standard deviation vectors.

**Note on Positional Encoding**: This implementation applies **Rotary Positional Embeddings (RoPE)** to the mean component ($\boldsymbol{\mu}$) within the attention layer to capture relative positions.

## 📂 Project Structure

```text
W2Attn/
├── w2attn/           # Core Package
│   ├── __init__.py
│   ├── attention.py    # W2Attention logic
│   ├── block.py        # W2TransformerBlock
│   ├── model.py        # LanguageModel wrapper
│   ├── config.py       # Configuration
│   ├── ffn.py          # FFN & RMSNorm
│   └── rope.py         # RotaryEmbedding implementation
├── benches/          # Benchmarks
├── tests/            # Tests
├── README.md
└── pyproject.toml
```

## 🛠️ Installation

Requires PyTorch, NumPy, and Einops:

```bash
uv add torch numpy einops
# or
pip install torch numpy einops
```

## 🚀 Quick Start

Building a model with `W2TransformerBlock`:

```python
import torch
from w2attn.block import W2TransformerBlock
from w2attn.rope import RotaryEmbedding

# 1. Configuration
class Config:
    hidden_size = 512
    num_attention_heads = 8
    intermediate_size = 2048
    rms_norm_eps = 1e-6

config = Config()
bs, seq_len = 2, 64

# 2. Initialize Module
block = W2TransformerBlock(config)
head_dim = config.hidden_size // config.num_attention_heads
rope = RotaryEmbedding(head_dim)

# 3. Input (Unified Stream)
# Hidden States (Batch, Seq, Hidden)
hidden_states = torch.randn(bs, seq_len, config.hidden_size)

# 4. Compute RoPE (Optional external computation for caching)
cos, sin = rope(hidden_states, seq_len=seq_len)

# 5. Forward Pass
out_hidden = block(
    hidden_states=hidden_states, 
    rotary_emb_outputs=(cos, sin)
)

print(f"Output Shape: {out_hidden.shape}")       # [2, 64, 512]
```

## ✅ Verification

Run the verification script to check shapes, forward pass, and gradients:

```bash
python tests/verify.py
```

Expected Output:
```text
Running Verification Tests...
Test 1: RoPE Shapes... PASSED
Test 2: Attention Forward... PASSED
Test 3: Block Forward... PASSED
Test 4: Gradients... PASSED
```

## 📝 Implementation Details

1.  **Distance Optimization**:
    $D_{\mu}^2$ is calculated using the expansion $\|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2 \mathbf{q}^T \mathcal{R}\mathbf{k}$ to utilize matrix multiplication.
2.  **Numerical Stability**:
    Attention scores are divided by $\tau + \epsilon$ to prevent division by zero. Sigma uses `Softplus` for non-negativity.
3.  **Architecture**:
    Interactions between mean and uncertainty occur only within the mixing process of the self-attention layer.

## 📊 Performance Analysis

Benchmarks (2025.12):

### 1. Associative Recall
W2 Attention performs excellently in tasks requiring fuzzy matching, showing **high parameter efficiency**.
*   **Result**: W2 saves ~43% parameters compared to Standard Attention with better convergence (Loss 3.45 vs 3.68).

### 2. Micro-Benchmarks (Memory)
With **Diagonal Sigma** optimization, memory usage is comparable to Standard Attention.
*   **Memory**: Reduced from GB-scale to MB-scale (e.g., 227MB for Seq=512), effectively matching Standard Attention.
*   **Speed**: ~1.2x slower than Standard Attention due to additional log/exp operations, but scaling remains $O(S^2)$.
