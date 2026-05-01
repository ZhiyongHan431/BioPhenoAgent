"""
CLI 入口 — 多 Agent 协作框架命令行工具

用法:
  python -m multi_agent_framework.cli run "需求描述"
  python -m multi_agent_framework.cli agent <agent_name> "任务描述"
  python -m multi_agent_framework.cli --config config.yaml run "需求描述"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
import traceback
from typing import Optional

from .core.config import Config
from .core.orchestrator import Orchestrator
from .agents.planning import PlanningAgent
from .agents.coding import CodingAgent
from .agents.data import DataAgent
from .agents.literature import LiteratureAgent
from .agents.ops import OpsAgent
from .core.message import Task, AgentRole, TaskStatus
from .utils.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-framework",
        description="多 Agent 协作框架 — 面向科研与工程自动化的 AI 代理系统",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径 (.yaml / .json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # run: 完整工作流
    run_parser = subparsers.add_parser("run", help="执行完整多 Agent 工作流")
    run_parser.add_argument("goal", type=str, help="需求描述")
    run_parser.add_argument("--output", "-o", type=str, help="输出结果到文件")

    # agent: 单 Agent 模式
    agent_parser = subparsers.add_parser("agent", help="调用单个 Agent")
    agent_parser.add_argument("role", type=str,
                              choices=["planning", "coding", "data", "literature", "ops"],
                              help="Agent 角色")
    agent_parser.add_argument("task", type=str, help="任务描述")
    agent_parser.add_argument("--output", "-o", type=str, help="输出结果到文件")

    # list: 列出可用 Agent
    subparsers.add_parser("list", help="列出所有可用 Agent")

    return parser


def create_orchestrator(config_path: Optional[str] = None) -> Orchestrator:
    if config_path and os.path.exists(config_path):
        config = Config.from_file(config_path)
    else:
        config = Config()

    orch = Orchestrator(config)

    orch.register_agent(PlanningAgent(config=config.agents.get("planning")))
    orch.register_agent(CodingAgent(config=config.agents.get("coding")))
    orch.register_agent(DataAgent(config=config.agents.get("data")))
    orch.register_agent(LiteratureAgent(config=config.agents.get("literature")))
    orch.register_agent(OpsAgent(config=config.agents.get("ops")))

    return orch


async def cmd_run(args: argparse.Namespace) -> int:
    orch = create_orchestrator(args.config)
    await orch.start()

    try:
        print(f"\n{'='*60}")
        print(f"  目标: {args.goal}")
        print(f"{'='*60}\n")

        result = await orch.run(args.goal)

        output = json.dumps(result, ensure_ascii=False, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\n结果已保存至: {args.output}")
        else:
            print(f"\n{'='*60}")
            print("  执行结果")
            print(f"{'='*60}")
            print(output)

        if result["status"] == "completed":
            return 0
        return 1

    finally:
        await orch.shutdown()


async def cmd_agent(args: argparse.Namespace) -> int:
    orch = create_orchestrator(args.config)

    role_map = {
        "planning": PlanningAgent,
        "coding": CodingAgent,
        "data": DataAgent,
        "literature": LiteratureAgent,
        "ops": OpsAgent,
    }

    agent_cls = role_map[args.role]
    agent = agent_cls(config=orch.config.agents.get(args.role))
    await agent.start()

    try:
        task = Task(title=f"单Agent任务: {args.role}", description=args.task)
        result_task = await agent.execute(task)

        output = json.dumps({
            "status": result_task.status.value,
            "result": result_task.result,
            "error": result_task.error,
        }, ensure_ascii=False, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"\n结果已保存至: {args.output}")
        else:
            print(f"\n{'='*60}")
            print(f"  [{args.role}] Agent 输出")
            print(f"{'='*60}")
            if result_task.result:
                print(result_task.result)
            if result_task.error:
                print(f"\n错误: {result_task.error}")

        return 0 if result_task.status == TaskStatus.COMPLETED else 1

    finally:
        await agent.shutdown()


def cmd_list(args: argparse.Namespace) -> int:
    agents_info = {
        "planning": "需求分析、任务拆解与工作流编排",
        "coding": "模型代码生成、训练脚本、Debug、图表绘制",
        "data": "数据清洗、Z-score归一化、特征工程与降噪",
        "literature": "论文精读、要点提取、综述框架、学术写作",
        "ops": "Linux环境配置、Docker打包、远程部署与持久化",
    }

    print(f"\n{'='*50}")
    print(f"  可用 Agent 列表")
    print(f"{'='*50}")
    for role, desc in agents_info.items():
        print(f"  {role:12s} | {desc}")
    print()
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    setup_logger(level=args.log_level if hasattr(args, "log_level") else "INFO")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "run":
            sys.exit(asyncio.run(cmd_run(args)))
        elif args.command == "agent":
            sys.exit(asyncio.run(cmd_agent(args)))
        elif args.command == "list":
            sys.exit(cmd_list(args))
        else:
            parser.print_help()
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
