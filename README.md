# W2Attn

[English](README_EN.md)

一个基于 PyTorch 实现的 **Wasserstein-2 Attention**。
本项目将 Transformer 中的传统点积注意力（Dot-Product Attention）替换为基于高斯分布之间 **Wasserstein-2 距离** 的度量。

## 📖 核心原理

在 W2 Attention 中，每一个 Token 被建模为一个对角高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$。为了兼容标准 Transformer 架构并节省显存，本实现采用 **统一残差流 (Unified Residual Stream)** 设计：

1.  **统一流**: 网络在层与层之间传递单一的隐藏状态向量 $\mathbf{h} \in \mathbb{R}^d$。
2.  **内部投影**: 在 W2 Attention 层内部，隐藏状态被投影并切分为 **均值 ($\boldsymbol{\mu}$)** 和 **不确定性 ($\boldsymbol{\sigma}$)** 分量。
3.  **W2 注意力计算**:
    注意力分数由两个分布间的 $W_2^2$ 距离决定：
    $$
    S_{m,n} = - \frac{D_{\mu}^2(m, n) + D_{\sigma}^2}{\tau}
    $$
    其中：
    -   $D_{\mu}^2$: 均值向量之间的欧氏距离（包含位置信息）。
    -   $D_{\sigma}^2$: 标准差向量之间的欧氏距离。

**关于位置编码**: 本实现将 **旋转位置编码 (RoPE)** 应用于注意力层内部的均值分量 ($\boldsymbol{\mu}$)，以捕捉相对位置信息。

## 📂 项目结构

```text
W2Attn/
├── w2attn/           # 核心包
│   ├── __init__.py
│   ├── attention.py    # W2Attention 逻辑
│   ├── block.py        # W2TransformerBlock
│   ├── model.py        # LanguageModel 包装器
│   ├── config.py       # 配置
│   ├── ffn.py          # FFN & RMSNorm
│   └── rope.py         # RotaryEmbedding 实现
├── benches/          # 基准测试
├── tests/            # 测试
├── README.md
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

使用 `W2TransformerBlock` 构建模型：

```python
import torch
from w2attn.block import W2TransformerBlock
from w2attn.rope import RotaryEmbedding

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

# 3. 输入 (统一流)
# Hidden States (Batch, Seq, Hidden)
hidden_states = torch.randn(bs, seq_len, config.hidden_size)

# 4. 计算 RoPE (可选外部计算以缓存)
cos, sin = rope(hidden_states, seq_len=seq_len)

# 5. 前向传播
out_hidden = block(
    hidden_states=hidden_states, 
    rotary_emb_outputs=(cos, sin)
)

print(f"Output Shape: {out_hidden.shape}")       # [2, 64, 512]
```

## ✅ 验证

运行验证脚本以检查形状、前向传播和梯度：

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
    在计算 $D_{\mu}^2$ 时，使用了展开公式 $\|\mathbf{q}\|^2 + \|\mathbf{k}\|^2 - 2 \mathbf{q}^T \mathcal{R}\mathbf{k}$ 以利用矩阵乘法。
2.  **数值稳定性**:
    Attention 分数除以 $\tau + \epsilon$ 防止除零错误。Sigma 使用 `Softplus` 激活保证非负性。
3.  **架构设计**:
    均值和不确定性的交互仅发生在自注意力层的混合过程中，随后被投影回统一流。

## 📊 性能分析

基准测试 (2025.12):

### 1. 关联记忆 (Associative Recall)
W2 Attention 在需要模糊匹配的任务中表现优异，显示出**极高的参数效率**。
*   **结果**: 相比 Standard Attention，W2 节省了 ~43% 的参数，并具有更好的收敛性 (Loss 3.45 vs 3.68)。

### 2. 微基准测试 (显存)
通过 **Diagonal Sigma** 优化，显存占用与 Standard Attention 相当。
*   **显存**: 从 GB 级别大幅降低至 MB 级别 (例如 Seq=512 时为 227MB)，有效匹配 Standard Attention。
*   **速度**: 比 Standard Attention 慢 ~1.2x（由于额外的 log/exp 操作），但复杂度增长仍保持为 $O(S^2)$。
