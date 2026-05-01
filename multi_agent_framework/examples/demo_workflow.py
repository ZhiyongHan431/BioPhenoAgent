#!/usr/bin/env python3
"""
示例脚本 — 演示多 Agent 协作框架的完整工作流

运行:
  cd multi_agent_framework/
  python examples/demo_workflow.py

输出:
  - 五个 Agent 的典型输出示例
  - 完整的任务拆解 → 分发 → 并行执行 → 整合流水线
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from multi_agent_framework.core.config import Config
from multi_agent_framework.core.orchestrator import Orchestrator
from multi_agent_framework.core.message import Task, TaskStatus, AgentRole
from multi_agent_framework.agents.planning import PlanningAgent
from multi_agent_framework.agents.coding import CodingAgent
from multi_agent_framework.agents.data import DataAgent
from multi_agent_framework.agents.literature import LiteratureAgent
from multi_agent_framework.agents.ops import OpsAgent


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def demo_single_agents():
    """演示各 Agent 独立工作"""
    agents = {
        "planning": PlanningAgent(),
        "coding": CodingAgent(),
        "data": DataAgent(),
        "literature": LiteratureAgent(),
        "ops": OpsAgent(),
    }

    tasks = {
        "planning": Task(
            title="科研项目任务拆解",
            description="使用 PyTorch 实现 GroupedPhenoVAE 模型，进行十万级样本多组学表型数据分析，并部署到远程 GPU 服务器",
            context={"task_type": "research"},
        ),
        "coding": Task(
            title="实现 VAE 模型代码",
            description="需要实现一个分组表型变分自编码器，支持分组特征编码、潜在空间正则化",
        ),
        "data": Task(
            title="数据加载与探索",
            description="加载表型数据 CSV 文件，进行基本探索性分析",
        ),
        "literature": Task(
            title="论文分析",
            description="分析关于 VAE 在生物表型组学中的应用的相关文献",
        ),
        "ops": Task(
            title="Docker 容器配置",
            description="配置深度学习 Docker 环境",
        ),
    }

    for name, agent in agents.items():
        await agent.start()
        try:
            print_section(f"Agent: {name}")
            result = await agent.execute(tasks[name])
            print(f"状态: {result.status.value}")
            if result.result:
                output = result.result
                if len(output) > 800:
                    output = output[:800] + "\n... (输出截断)"
                print(output)
            if result.error:
                print(f"错误: {result.error}")
        finally:
            await agent.shutdown()


async def demo_full_workflow():
    """演示完整的多 Agent 协作工作流"""
    print_section("启动多 Agent 编排引擎")

    config = Config(workspace_dir="./demo_workspace")
    orch = Orchestrator(config)

    orch.register_agent(PlanningAgent())
    orch.register_agent(CodingAgent())
    orch.register_agent(DataAgent())
    orch.register_agent(LiteratureAgent())
    orch.register_agent(OpsAgent())

    await orch.start()

    try:
        goal = """
        科研任务：开发 GroupedPhenoVAE 模型用于多组学生物表型数据分析。
        需要：
        1. 编写 VAE 模型代码
        2. 数据预处理流水线（数据清洗、Z-score归一化、特征降噪）
        3. 训练脚本开发
        4. 结果可视化图表
        """

        print_section("工作流目标")
        print(f"  {goal.strip()}")

        result = await orch.run(goal)

        print_section("工作流结果")
        print(f"  状态: {result['status']}")

        if result.get("task_tree"):
            task_tree = result["task_tree"]
            print(f"\n  根任务: {task_tree['title']}")
            for i, sub in enumerate(task_tree.get("subtasks", []), 1):
                print(f"    子任务 {i}: [{sub['assigned_to']}] {sub['title']} ({sub['status']})")

        print(f"\n  整合结果:")
        print(json.dumps(result.get("results", {}), ensure_ascii=False, indent=4))

    finally:
        await orch.shutdown()


def main():
    parser = argparse.ArgumentParser(description="多 Agent 框架演示")
    parser.add_argument("--full", action="store_true", help="运行完整工作流演示")
    parser.add_argument("--single", action="store_true", help="运行单 Agent 演示")
    args = parser.parse_args()

    # 默认两者都运行
    run_full = args.full or not args.single
    run_single = args.single or not args.full

    if run_single:
        print("\n" + "="*60)
        print("  演示 1: 各 Agent 独立输出")
        print("="*60)
        asyncio.run(demo_single_agents())

    if run_full:
        print("\n" + "="*60)
        print("  演示 2: 完整多 Agent 协作工作流")
        print("="*60)
        asyncio.run(demo_full_workflow())

    print_section("演示完成")
    print("  框架已就绪。配置 LLM API Key 后可启用完整大模型推理能力。")
    print("  编辑 config.yaml 设置 api_key 后即可接入真实 LLM。\n")


if __name__ == "__main__":
    main()
