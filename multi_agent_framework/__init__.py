"""
多 Agent 协作框架 — 面向科研与工程自动化工作流的 AI 代理系统

五大协同模块:
  - PlanningAgent:   需求拆解、任务拆分与工作流编排
  - CodingAgent:     模型架构编写、训练脚本开发、自动 Debug 与算法迭代
  - DataAgent:       原始数据清洗、Z-score 归一化、Log 变换、特征降噪
  - LiteratureAgent: 论文精读、要点提取、综述框架生成
  - OpsAgent:        Linux 服务器配置、Docker 容器打包、远程部署

架构: 长链推理 + 多 Agent 协作，基于消息传递的异步编排引擎
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent Framework Team"
