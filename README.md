# W2-RoPE Transformer

一个基于 PyTorch 实现的 **Wasserstein-2 Attention with RoPE** 架构。
该模型将 Transformer 中的传统点积注意力（Dot-Product Attention）替换为基于高斯分布之间 **Wasserstein-2 距离** 的度量，并针对均值流融入了 **旋转位置编码 (RoPE)**。

## 📖 核心原理

虽然在注意力机制内部每一个 Token 被建模为一个对角高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，但为了兼容标准 Transformer 架构并节省显存，本实现采用 **统一残差流 (Unified Residual Stream)** 设计：

1.  **统一流**: 网络在层与层之间传递单一的隐藏状态向量 $\mathbf{h} \in \mathbb{R}^d$。
2.  **内部投影**: 在 W2 Attention 层内部，隐藏状态被投影并切分为 **均值 ($\boldsymbol{\mu}$)** 和 **不确定性 ($\boldsymbol{\sigma}$)** 分量。
3.  **W2 注意力计算**:
    注意力分数由两个分布间的 $W_2^2$ 距离决定：
    $$
    S_{m,n} = - \frac{D_{\mu}^2(m, n) + D_{\sigma}^2}{\tau}
    $$
    其中：
    -   **位置项** $D_{\mu}^2$: 通过 RoPE 旋转后的向量欧氏距离计算。
    -   **形态项** $D_{\sigma}^2$: 标准差向量之间的欧氏距离。

### 旋转位置编码 (RoPE)
RoPE 仅应用于注意力层内部的 **均值分量** ($\boldsymbol{\mu}$)，保持不确定性分量的位置不变性。这使得模型能够捕捉绝对和相对位置信息。

## 📂 项目结构

```text
W2Attn/
├── w2_rope/
│   ├── __init__.py     # Exports
│   ├── attention.py    # W2Attention & StandardAttention
│   ├── rope.py         # RotaryEmbedding
│   ├── ffn.py          # FeedForward & RMSNorm
│   ├── block.py        # W2TransformerBlock & StandardBlock
│   ├── config.py       # ModelConfig
│   └── model.py        # LanguageModel (Unified Model Wrapper)
├── benches/
│   ├── run_benchmarks.py # Unified Benchmark Entry Point
│   ├── common.py         # Shared Benchmark Utils
│   └── hierarchy.py      # Entailment Data Generator
├── tests/
│   ├── test_components.py
│   └── verify.py
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

如何使用 `W2TransformerBlock` 构建模型（注意：接口采用统一流输入）：

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

# 3. 准备输入 (统一流)
# Hidden States (Batch, Seq, Hidden)
hidden_states = torch.randn(bs, seq_len, config.hidden_size)

# 4. 计算 RoPE 旋转项
# 需在外部计算并传入，以便在多层间共享或缓存
# RoPE 仅需基于序列长度计算一次
cos, sin = rope(hidden_states, seq_len=seq_len)

# 5. 前向传播
out_hidden = block(
    hidden_states=hidden_states, 
    rotary_emb_outputs=(cos, sin)
)

print(f"Output Shape: {out_hidden.shape}")       # [2, 64, 512]
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
    Attention 分数除以 $\tau + \epsilon$ 防止除零错误。
    Sigma 激活使用了 `Softplus` 以保证非负性。
3.  **FFN**:
    使用标准的 SwiGLU FFN 处理统一的隐藏状态。均值和不确定性的交互发生在自注意力层的混合过程中，随后被投影回统一流。

## 📊 性能分析 (Performance Analysis)

基于 `benches/run_benchmarks.py` 的测试结果 (2025.12):

### 1. 关联记忆 (Associative Recall) —— 强项
W2 Attention 在需要模糊匹配和记忆的任务中表现优异，**参数利用率极高**。

| 模型 | 参数量 | Loss | 备注 |
| :--- | :--- | :--- | :--- |
| **Standard Attention** | 492k | 3.68 | Baseline |
| **W2 Attention** | **279k** | **3.45** | **更少参数，更低 Loss** |

*   **结论**: W2 节省了 ~43% 的参数，却取得了更好的收敛效果。

### 2. 逻辑推理 (Entailment) —— 改进
优化后的 W2 Attention 在逻辑推理任务上表现与 Standard Attention 持平，能够达到 100% 准确率。

*   **Standard**: Accuracy 100%, Loss 0.0078
*   **W2**: **Accuracy 100%, Loss 0.0067**

*   **结论**: 采用 Scalar Sigma 近似后，模型不仅大幅节省显存，且保留了原本的学习能力。

### 3. 微基准测试 (Micro-Benchmarks) —— 显存优化
经过 `dist_sq` 展开与 Scalar Sigma 优化后，W2 Attention 的 **显存占用** 已大幅降低，彻底消除了显存爆炸问题。

| 实验场景 | W2 Loss | Std Loss | W2 显存 | Std 显存 | W2 速度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base (Seq=128) | **6.80** | 6.94 | **198MB** | 192MB | ~1.2x Slower |
| Long (Seq=512) | **6.94** | 6.96 | **227MB** | 176MB | ~1.2x Slower |
| Deep (L=4) | **5.84** | 6.49 | **311MB** | 300MB | ~1.2x Slower |

*   **显存优化**: 显存占用从原先的 GB 级别（如 Long 场景 2.9GB）降低至 MB 级别（227MB），与 Standard Attention 几乎持平。
*   **速度**: 虽然引入了额外的 log 和 exp 计算，但避免了大张量读写，速度与 Standard Attention 相当。
*   **实现**: 使用了 $\mathbf{q}^2 + \mathbf{k}^2 - 2\mathbf{q}\mathbf{k}^T$ 展开和 `sigma` 标量近似，将复杂度从 $O(S^2 D)$ 降低为 $O(S^2)$。
