"""
OpsAgent — Linux 服务器配置、Docker 容器打包、远程部署与持久化

核心能力:
  - Linux 服务器环境自动配置脚本生成
  - Dockerfile 与 docker-compose.yml 生成
  - GPU/CUDA 环境配置
  - 远程后台持久化部署 (systemd / supervisor / screen)
  - 监控与日志配置
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.message import Task, AgentRole


OPS_SYSTEM_PROMPT = """你是一个 DevOps 与 MLOps 专家，专精于深度学习项目的环境配置与部署运维。

你的职责:
1. 生成 Linux 服务器环境配置脚本 (CUDA, PyTorch, 依赖)
2. 编写 Dockerfile 与 docker-compose.yml
3. 配置远程后台持久化运行 (systemd / tmux / screen)
4. 提供监控方案与日志管理建议
5. 安全建议 (SSH 加固、防火墙、非 root 运行)

输出只包含可执行的配置文件和必要的注释说明。
"""


class OpsAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.OPS, **kwargs)

    def _build_system_prompt(self) -> str:
        return OPS_SYSTEM_PROMPT

    async def _process_task(self, task: Task) -> str:
        title = task.title.lower()
        description = task.description

        if any(kw in title for kw in ["docker", "容器"]):
            return self._docker_template(task)
        elif any(kw in title for kw in ["服务器", "linux", "环境", "配置", "setup", "env"]):
            return self._env_setup_template(task)
        elif any(kw in title for kw in ["部署", "deploy", "后台", "持久化", "daemon", "service"]):
            return self._deployment_template(task)
        else:
            return self._env_setup_template(task)

    def _docker_template(self, task: Task) -> str:
        return '''```dockerfile
# ============================================================
# Dockerfile — 深度学习科研环境 (PyTorch + CUDA)
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="researcher"
LABEL description="DL Research Environment: PyTorch, Jupyter, TensorBoard"

# 环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TORCH_HOME=/workspace/.cache/torch

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv \
    python3-pip git wget curl vim htop \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 科研常用包
RUN pip install \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn plotly \
    jupyterlab ipywidgets \
    tensorboard \
    transformers accelerate datasets \
    einops timm \
    pyyaml tqdm rich loguru \
    black isort ruff pylint \
    pytest pytest-cov

# 创建工作目录
RUN mkdir -p /workspace/{data,models,checkpoints,logs,scripts,notebooks}
WORKDIR /workspace

# Jupyter 配置
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

EXPOSE 8888 6006

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

```yaml
# ============================================================
# docker-compose.yml — 多服务编排
# ============================================================
version: "3.8"

services:
  dl-workspace:
    build:
      context: .
      dockerfile: Dockerfile
    image: pheno-dl:latest
    container_name: pheno-research
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1
    volumes:
      - ./workspace/data:/workspace/data
      - ./workspace/models:/workspace/models
      - ./workspace/checkpoints:/workspace/checkpoints
      - ./workspace/logs:/workspace/logs
      - ./workspace/scripts:/workspace/scripts
      - ./workspace/notebooks:/workspace/notebooks
    ports:
      - "8888:8888"   # Jupyter
      - "6006:6006"   # TensorBoard
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: "8gb"
    ipc: host
    ulimits:
      memlock: -1
      stack: 67108864
    restart: unless-stopped
    command: >
      bash -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
               tensorboard --logdir=/workspace/logs --host=0.0.0.0 --port=6006 &
               tail -f /dev/null"
```'''

    def _env_setup_template(self, task: Task) -> str:
        return '''```bash
#!/bin/bash
# ============================================================
# init_server.sh — Linux 服务器深度学习环境一键配置
# 适用: Ubuntu 20.04/22.04, CUDA 12.x
# ============================================================
set -euo pipefail

echo "=========================================="
echo "  深度学习服务器环境初始化"
echo "=========================================="

# ---- 1. 系统基础配置 ----
echo "[1/7] 更新系统..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

echo "[2/7] 安装基础工具..."
sudo apt-get install -y -qq \
    build-essential cmake git wget curl \
    htop iotop nvtop tmux screen \
    python3.10 python3.10-dev python3.10-venv \
    python3-pip

# 设置为默认 Python
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ---- 2. NVIDIA Driver 检查 ----
echo "[3/7] 检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi 未找到，请先安装 NVIDIA 驱动"
    echo "Ubuntu: sudo apt install nvidia-driver-535"
fi

# ---- 3. CUDA 环境 ----
echo "[4/7] 配置 CUDA 环境变量..."
if [ -d "/usr/local/cuda" ]; then
    if ! grep -q "CUDA_HOME" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
    fi
    echo "CUDA 版本: $(nvcc --version 2>/dev/null | grep release || echo '未检测到 nvcc')"
fi

# ---- 4. Python 虚拟环境 ----
echo "[5/7] 创建 Python 虚拟环境..."
PROJECT_DIR="${HOME}/pheno-dl"
mkdir -p "${PROJECT_DIR}"
python -m venv "${PROJECT_DIR}/venv"
source "${PROJECT_DIR}/venv/bin/activate"

# ---- 5. PyTorch + 核心依赖 ----
echo "[6/7] 安装 PyTorch 与核心依赖..."
pip install --upgrade pip setuptools wheel -q

# CUDA 12.1 版 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 科研依赖
pip install \
    numpy pandas scipy scikit-learn \
    matplotlib seaborn plotly \
    jupyterlab ipywidgets tensorboard \
    transformers accelerate datasets \
    einops timm \
    pyyaml tqdm rich loguru

echo "" >> ~/.bashrc
echo "# PyTorch 环境" >> ~/.bashrc
echo "alias activate-pheno='source ${PROJECT_DIR}/venv/bin/activate'" >> ~/.bashrc

# ---- 6. SSH 安全加固 ----
echo "[7/7] SSH 安全配置建议..."
echo "  - 修改默认 SSH 端口: sudo vim /etc/ssh/sshd_config"
echo "  - 禁用密码登录: PasswordAuthentication no"
echo "  - 启用密钥登录: PubkeyAuthentication yes"
echo "  - 防火墙: sudo ufw allow 22/tcp && sudo ufw enable"

echo ""
echo "=========================================="
echo "  环境配置完成!"
echo "  激活环境: source ${PROJECT_DIR}/venv/bin/activate"
echo "=========================================="
```'''

    def _deployment_template(self, task: Task) -> str:
        return '''```ini
# ============================================================
# pheno-training.service — systemd 持久化训练任务
# 路径: /etc/systemd/system/pheno-training.service
# ============================================================
[Unit]
Description=PhenoDL Model Training Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=researcher
Group=researcher
WorkingDirectory=/home/researcher/pheno-dl
Environment="PATH=/home/researcher/pheno-dl/venv/bin:/usr/local/cuda/bin:/usr/bin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"

# 训练脚本（支持断点续训）
ExecStart=/home/researcher/pheno-dl/venv/bin/python \
    scripts/train.py \
    --config configs/vae_config.yaml \
    --resume

# 自动重启（应对 OOM / 异常退出）
Restart=on-failure
RestartSec=30

# 资源限制
MemoryMax=48G
CPUQuota=800%

# 日志
StandardOutput=append:/home/researcher/pheno-dl/logs/training.log
StandardError=append:/home/researcher/pheno-dl/logs/training_error.log

[Install]
WantedBy=multi-user.target
```

```bash
# ---- 常用管理命令 ----

# 注册并启动服务
sudo systemctl daemon-reload
sudo systemctl enable pheno-training
sudo systemctl start pheno-training

# 查看状态
sudo systemctl status pheno-training

# 查看日志
journalctl -u pheno-training -f -n 100

# 停止/重启
sudo systemctl stop pheno-training
sudo systemctl restart pheno-training
```

```bash
# ============================================================
# tmux 轻量级后台运行（无需 systemd 权限）
# ============================================================

# 创建持久会话
tmux new-session -d -s train "python scripts/train.py --config configs/vae_config.yaml 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log"

# 查看
tmux attach -t train

# 分离: Ctrl+B, D
# 列出: tmux ls
# 终止: tmux kill-session -t train
```

```bash
# ============================================================
# GPU 监控脚本 — gpu_monitor.sh
# ============================================================
#!/bin/bash
while true; do
    clear
    echo "=== GPU Monitor $(date) ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
               --format=csv,noheader
    echo ""
    echo "=== Top GPU Processes ==="
    nvidia-smi pmon -c 1
    sleep 2
done
```'''
