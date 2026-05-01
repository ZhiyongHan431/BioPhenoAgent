"""
PlanningAgent — 需求分析、任务拆解与工作流编排

核心能力:
  - 接收用户复杂需求，拆解为层级化子任务树
  - 为各子任务标注所需 Agent 类型、优先级、依赖关系
  - 输出结构化 JSON 任务树，供 Orchestrator 分发
"""

from __future__ import annotations

import json
from typing import Any

from ..core.agent import BaseAgent
from ..core.message import Task, AgentRole


PLANNING_SYSTEM_PROMPT = """你是一个科研/工程任务规划专家。你的职责是将用户的复杂需求拆解为层级化的可执行子任务。

规则:
1. 输出严格的 JSON 格式，不要包含任何其他文字
2. 任务树的根节点描述总体目标
3. 每个叶子任务对应一个专职 Agent（coding / data / literature / ops）
4. 每层不超过 7 个子任务
5. 为每个任务标注 assigned_to（Agent 角色类型）

输出 JSON 结构:
{
  "title": "...",
  "description": "...",
  "assigned_to": "planning",
  "subtasks": [
    {
      "title": "...",
      "description": "...",
      "assigned_to": "coding|data|literature|ops",
      "tags": ["scikit-learn", "classification"],
      "subtasks": []
    }
  ]
}

assigned_to 选择规则:
- 涉及代码编写、模型实现、训练脚本 → coding
- 涉及数据清洗、归一化、特征工程 → data
- 涉及论文分析、文献综述、学术写作 → literature
- 涉及服务器部署、Docker、环境配置 → ops

请输出 JSON:"""


class PlanningAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.PLANNING, **kwargs)

    def _build_system_prompt(self) -> str:
        return PLANNING_SYSTEM_PROMPT

    async def _process_task(self, task: Task) -> str:
        goal = task.description or task.context.get("raw_goal", "")
        keywords = task.context.get("keywords", [])
        task_type = task.context.get("task_type", "general")

        # 基于关键词和任务类型的规则推理解析
        plan = self._rule_based_plan(goal, keywords, task_type)
        return json.dumps(plan, ensure_ascii=False, indent=2)

    def _rule_based_plan(self, goal: str, keywords: list[str], task_type: str) -> dict:
        """基于规则的快速拆解（备选方案，当无 LLM 时使用）"""
        subtasks = []

        # 数据处理相关
        data_keywords = ["数据", "表型", "组学", "归一化", "清洗", "特征", "预处理", "csv", "excel"]
        if any(kw in goal for kw in data_keywords):
            subtasks.append({
                "title": "数据加载与初步探索",
                "description": "加载原始数据文件，输出基本信息（行列数、缺失值、数据类型、描述性统计）",
                "assigned_to": "data",
                "tags": ["eda", "data-loading"],
                "subtasks": [],
            })
            subtasks.append({
                "title": "数据清洗与标准化",
                "description": "处理缺失值、异常值，执行 Z-score 归一化与 Log 变换",
                "assigned_to": "data",
                "tags": ["cleaning", "normalization"],
                "subtasks": [],
            })

        # 模型相关
        model_keywords = ["模型", "vae", "transformer", "moe", "网络", "训练", "pytorch", "深度学习"]
        if any(kw in goal.lower() for kw in model_keywords):
            subtasks.append({
                "title": "模型架构设计与实现",
                "description": f"设计并实现模型架构，包含前向传播与损失函数定义。目标: {goal[:200]}",
                "assigned_to": "coding",
                "tags": ["model-architecture", "deep-learning"],
                "subtasks": [],
            })
            subtasks.append({
                "title": "训练脚本开发",
                "description": "编写训练循环脚本，包含数据加载器、优化器配置、学习率调度、checkpoint 保存与 TensorBoard 日志",
                "assigned_to": "coding",
                "tags": ["training", "script"],
                "subtasks": [],
            })

        # 论文相关
        paper_keywords = ["论文", "文献", "综述", "摘要", "精读", "sci"]
        if any(kw in goal for kw in paper_keywords):
            subtasks.append({
                "title": "文献检索与分析",
                "description": "根据主题检索相关文献，提取关键方法、实验结果与创新点",
                "assigned_to": "literature",
                "tags": ["paper-review", "literature"],
                "subtasks": [],
            })

        # 部署相关
        deploy_keywords = ["部署", "docker", "服务器", "linux", "环境", "容器"]
        if any(kw in goal for kw in deploy_keywords):
            subtasks.append({
                "title": "部署环境配置",
                "description": "配置 Linux 服务器环境、编写 Dockerfile、构建容器镜像",
                "assigned_to": "ops",
                "tags": ["deployment", "docker"],
                "subtasks": [],
            })

        # 图表相关
        viz_keywords = ["图", "可视化", "绘图", "图表", "plot", "figure"]
        if any(kw in goal.lower() for kw in viz_keywords):
            subtasks.append({
                "title": "科研图表绘制",
                "description": "使用 matplotlib/seaborn 生成符合 SCI 发表标准的图表",
                "assigned_to": "coding",
                "tags": ["visualization", "figures"],
                "subtasks": [],
            })

        if not subtasks:
            subtasks.append({
                "title": "任务执行",
                "description": goal,
                "assigned_to": "coding",
                "tags": ["general"],
                "subtasks": [],
            })

        return {
            "title": f"项目: {goal[:80]}",
            "description": goal,
            "assigned_to": "planning",
            "tags": keywords,
            "subtasks": subtasks,
        }
