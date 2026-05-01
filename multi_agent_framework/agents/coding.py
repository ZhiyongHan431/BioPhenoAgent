"""
CodingAgent — 代码生成、模型架构实现、训练脚本开发、自动 Debug

核心能力:
  - 深度学习模型代码编写 (VAE, Transformer, MoE, CNN, etc.)
  - 训练/评估脚本生成
  - 自动错误检测与修复建议
  - 科研图表绘制代码
  - 代码重构与优化
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.message import Task, AgentRole


CODING_SYSTEM_PROMPT = """你是一个资深 AI/ML 代码工程师，专注于科研深度学习模型开发。

你的职责:
1. 生成生产级 Python 代码，遵循 PEP8 规范
2. 使用 PyTorch / TensorFlow 等主流框架
3. 包含完整的类型注解、文档字符串、错误处理
4. 提供训练脚本、评估脚本、推理脚本
5. 遇到 bug 时提供诊断和修复方案

输出只包含代码和必要的简要说明，代码块用 ```python``` 包裹。

当任务是调试时，请:
1. 先分析错误原因
2. 再给出修复后的完整代码
3. 说明修改了什么以及为什么
"""


class CodingAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.CODING, **kwargs)

    def _build_system_prompt(self) -> str:
        return CODING_SYSTEM_PROMPT

    async def _process_task(self, task: Task) -> str:
        title = task.title.lower()
        description = task.description

        if any(kw in title for kw in ["vae", "variational autoencoder", "变分自编码器"]):
            return self._generate_vae_template(task)
        elif any(kw in title for kw in ["transformer", "注意力"]):
            return self._generate_transformer_template(task)
        elif any(kw in title for kw in ["moe", "mixture of experts", "混合专家"]):
            return self._generate_moe_template(task)
        elif any(kw in title for kw in ["训练脚本", "training", "train"]):
            return self._generate_training_script(task)
        elif any(kw in title for kw in ["图", "可视化", "绘图", "plot", "chart", "figure"]):
            return self._generate_plot_template(task)
        elif any(kw in title for kw in ["调试", "debug", "修复", "bug", "fix", "error"]):
            return self._debug_response(task)
        else:
            return self._generate_general_code(task)

    # ---- 模型模板生成 ----

    def _generate_vae_template(self, task: Task) -> str:
        """生成 VAE 模型模板"""
        return '''```python
"""
GroupedPhenoVAE — 面向分组表型数据的变分自编码器
支持分组特征编码、潜在空间正则化、双向重构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GroupedPhenoVAE(nn.Module):
    """分组表型 VAE: 按特征组独立编码后融合"""

    def __init__(
        self,
        group_dims: list[int],  # 各组特征维度
        latent_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_dims = group_dims
        self.latent_dim = latent_dim

        # 分组编码器
        self.group_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            ) for d in group_dims
        ])

        # 融合层
        fusion_input = (hidden_dim // 2) * len(group_dims)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 潜在空间
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, sum(group_dims)),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        splits = torch.split(x, self.group_dims, dim=1)
        encoded = [enc(s) for enc, s in zip(self.group_encoders, splits)]
        fused = self.fusion(torch.cat(encoded, dim=1))
        return self.mu(fused), self.logvar(fused)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total = recon_loss + beta * kl_loss
        return total, recon_loss, kl_loss
```'''

    def _generate_transformer_template(self, task: Task) -> str:
        return '''```python
"""
PhenoFormer — 面向表型序列数据的轻量 Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class PhenoFormer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, n_features)
        self.classifier = nn.Linear(d_model, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        outputs = {"reconstruction": self.output_proj(encoded)}
        if self.classifier is not None:
            outputs["logits"] = self.classifier(encoded.mean(dim=1))
        return outputs
```'''

    def _generate_moe_template(self, task: Task) -> str:
        return '''```python
"""
PhenoMoE — 混合专家模型用于表型多组学数据集成
支持 Sparse Gating 与 Load Balancing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SparseMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        expert_hidden: int = 256,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, output_dim),
            ) for _ in range(num_experts)
        ])

        self.router = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim = x.shape

        router_logits = self.router(x)  # [B, E]
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        # 容量限制
        capacity = max(1, int(self.capacity_factor * batch_size / self.num_experts))

        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)

        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx).any(dim=-1)
            if mask.sum() > capacity:
                indices = mask.nonzero(as_tuple=True)[0][:capacity]
                mask = torch.zeros_like(mask)
                mask[indices] = True

            if mask.sum() > 0:
                expert_input = x[mask]
                expert_output = self.experts[expert_idx](expert_input)
                gate_mask = (top_k_indices[mask] == expert_idx).float()
                gates = (top_k_gates[mask] * gate_mask).sum(dim=-1, keepdim=True)
                output[mask] += gates * expert_output

        # Load Balancing Loss
        expert_load = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            expert_load[i] = (top_k_indices == i).sum().float()
        load_balance_loss = expert_load.std() / (expert_load.mean() + 1e-8)

        return output, load_balance_loss
```'''

    def _generate_training_script(self, task: Task) -> str:
        return '''```python
"""
通用训练脚本 — 支持 VAE / Transformer / MoE 模型训练
包含: 数据加载、训练循环、验证、早停、Checkpoint、TensorBoard 日志
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Optional, Callable
import json
from datetime import datetime


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",
        early_stopping_patience: int = 15,
        grad_clip: float = 1.0,
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.log_interval = log_interval

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.save_dir / f"runs_{ts}")

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = early_stopping_patience
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            data = batch[0].to(self.device)
            self.optimizer.zero_grad()

            recon, mu, logvar = self.model(data)
            loss = self.model.loss_function(recon, data, mu, logvar)
            if isinstance(loss, tuple):
                loss = loss[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.log_interval == 0:
                self.writer.add_scalar("batch/loss", loss.item(), batch_idx)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        if not self.val_loader:
            return 0.0
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            data = batch[0].to(self.device)
            recon, mu, logvar = self.model(data)
            loss = self.model.loss_function(recon, data, mu, logvar)
            if isinstance(loss, tuple):
                loss = loss[0]
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self) -> dict:
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.writer.add_scalars("epoch/loss", {
                "train": train_loss, "val": val_loss
            }, epoch)
            self.writer.add_scalar("epoch/lr", self.optimizer.param_groups[0]["lr"], epoch)

            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }, self.save_dir / "best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history
```'''

    def _generate_plot_template(self, task: Task) -> str:
        return '''```python
"""
科研图表绘制 — 面向 SCI 论文的可视化模板
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def plot_training_curves(
    history: dict,
    save_path: str = "training_curves.pdf",
    title: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], color="#2E86AB", linewidth=1.5, label="Train")
    axes[0].plot(history["val_loss"], color="#A23B72", linewidth=1.5, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend(frameon=False)

    if "lr" in history:
        axes[1].plot(history["lr"], color="#F18F01", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")

    if title:
        fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return fig


def plot_latent_space(
    z: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: str = "latent_space.pdf",
    method: str = "tsne",
):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    reducer = TSNE(n_components=2, random_state=42) if method == "tsne" \
              else PCA(n_components=2, random_state=42)

    z_2d = reducer.fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 7))

    if labels is not None:
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap="viridis",
                            alpha=0.7, s=15, edgecolors="none")
        plt.colorbar(scatter, ax=ax, label="Group")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, s=15, color="#2E86AB",
                   edgecolors="none")

    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_title("Latent Space Visualization")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str = "feature_importance.pdf",
):
    indices = np.argsort(importance)[-top_n:]
    values = importance[indices]
    names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 2))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))

    ax.barh(range(top_n), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return fig
```'''

    def _debug_response(self, task: Task) -> str:
        return f"""## Debug Analysis for: {task.title}

**Common PyTorch issues and fixes:**

1. **CUDA out of memory** → Reduce batch size, use gradient accumulation,
   or add `torch.cuda.empty_cache()`

2. **Shape mismatch** → Check:
   ```python
   print(f"Input: {{x.shape}}, Expected: {{expected_shape}}")
   ```
   - Ensure encoder output matches decoder input
   - Verify data loader produces correct dimensions

3. **NaN loss** → Check for:
   - `log(0)` or `sqrt(negative)` in loss functions
   - Exploding gradients (add gradient clipping)
   - Learning rate too high

4. **Model not learning** → Verify:
   - Data normalization is correct
   - Loss function gradients exist
   - Optimizer parameters include all trainable params

**For your specific issue**, please provide:
- Full error traceback
- Model input shapes
- Training configuration"""

    def _generate_general_code(self, task: Task) -> str:
        return f'''```python
"""
Auto-generated code stub for: {task.title}
"""
from typing import Any


def solve_{task.title.lower().replace(" ", "_")[:50]}(
    *args: Any, **kwargs: Any
) -> Any:
    \"\"\"
    Task: {task.title}
    Description: {task.description[:200]}
    \"\"\"
    raise NotImplementedError(
        "Please use an LLM backend for complex code generation. "
        "Set up your API key in the config and retry."
    )
```'''
