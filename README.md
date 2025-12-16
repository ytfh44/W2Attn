# W2-RoPE Transformer

一个基于 PyTorch 实现的 **Wasserstein-2 Attention with RoPE** 架构。
该模型将 Transformer 中的传统点积注意力（Dot-Product Attention）替换为基于高斯分布之间 **Wasserstein-2 距离** 的度量，并针对均值流融入了 **旋转位置编码 (RoPE)**。

## 📖 核心原理

每一个 Token 被建模为一个对角高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$。网络维护两个独立的残差流：
1.  **均值流 (Mean Stream, $\boldsymbol{\mu}$)**: 代表特征的中心位置。
2.  **不确定性流 (Uncertainty Stream, $\mathbf{l} = \log \boldsymbol{\Sigma}$)**: 代表特征的不确定性（方差的对数）。

### 1. Wasserstein-2 注意力
注意力分数由两个分布间的 $W_2^2$ 距离决定：
$$
S_{m,n} = - \frac{D_{\mu}^2(m, n) + D_{\sigma}^2}{\tau}
$$
其中：
-   **位置项** $D_{\mu}^2$: 通过 RoPE 旋转后的向量欧氏距离计算。
-   **形态项** $D_{\sigma}^2$: 标准差向量之间的欧氏距离。

### 2. 旋转位置编码 (RoPE)
RoPE 仅应用于 **均值流** ($\boldsymbol{\mu}$)，保持不确定性流的位置不变性。这使得模型能够捕捉绝对和相对位置信息，同时处理分布的几何特性。

### 3. 输出聚合
-   **均值输出**: 值的加权算术平均。
-   **不确定性输出**: 值的加权几何平均（对数域的算术平均），反映了不确定性的聚合。

## 📂 项目结构

```text
d:/PROJECTS/W2Attn/
├── w2_rope/
│   ├── __init__.py
│   ├── attention.py    # W2Attention 核心实现 (W2 距离计算, 双流投影)
│   ├── rope.py         # RotaryEmbedding 实现 (仅作用于均值)
│   ├── ffn.py          # W2FeedForward (双流 SwiGLU) 和 RMSNorm
│   └── block.py        # W2TransformerBlock (完整的 Transformer 块)
├── tests/
│   ├── verify.py       # 手动验证脚本 (Shapes, Forward, Gradients)
│   └── test_components.py # Pytest 测试用例
├── README.md           # 项目文档
└── pyproject.toml
```

## 🛠️ 安装

需要安装 PyTorch, NumPy 和 Einops：

```bash
uv add torch numpy einops
# 或者
pip install torch numpy einops
```

## 🚀 快速开始

如何使用 `W2TransformerBlock` 构建模型：

```python
import torch
from w2_rope.block import W2TransformerBlock
from w2_rope.rope import RotaryEmbedding

# 1. 配置参数
class Config:
    hidden_size = 512
    num_attention_heads = 8
    intermediate_size = 2048
    rms_norm_eps = 1e-6

config = Config()
bs, seq_len = 2, 64

# 2. 初始化模块
block = W2TransformerBlock(config)
head_dim = config.hidden_size // config.num_attention_heads
rope = RotaryEmbedding(head_dim)

# 3. 准备输入 (双流)
# 均值流 (Batch, Seq, Hidden)
mu = torch.randn(bs, seq_len, config.hidden_size)
# 不确定性流 (log sigma)
log_sigma = torch.randn(bs, seq_len, config.hidden_size)

# 4. 计算 RoPE 旋转项
# 需在外部计算并传入，以便在多层间共享或缓存
cos, sin = rope(mu, seq_len=seq_len)

# 5. 前向传播
out_mu, out_log_sigma = block(
    hidden_states_mu=mu, 
    hidden_states_log_sigma=log_sigma, 
    rotary_emb_outputs=(cos, sin)
)

print(f"Output Mean Shape: {out_mu.shape}")       # [2, 64, 512]
print(f"Output Sigma Shape: {out_log_sigma.shape}") # [2, 64, 512]
```

## ✅ 验证

项目中包含一个验证脚本，用于检查形状正确性、前向传播和梯度反向传播。

```bash
python tests/verify.py
```

预期输出：
```text
Running Verification Tests...
Test 1: RoPE Shapes... PASSED
Test 2: Attention Forward... PASSED
Test 3: Block Forward... PASSED
Test 4: Gradients... PASSED
```

## 📝 实现细节备忘

1.  **距离计算优化**:
    在计算 $D_{\mu}^2$ 时，使用了展开公式 $\|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2 \mathbf{q}^T \mathcal{R}\mathbf{k}$ 以充分利用矩阵乘法加速。
2.  **数值稳定性**:
    Attention 分署除以 $\tau + \epsilon$ 防止除零错误。
3.  **双流 FFN**:
    使用 SwiGLU 激活函数，均值流和不确定性流拥有独立的权重参数，互如果不干扰。
