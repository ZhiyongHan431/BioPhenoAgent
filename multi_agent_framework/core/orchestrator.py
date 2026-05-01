"""
Orchestrator — 多 Agent 编排引擎

负责:
  1. 接收用户需求 → 路由至 PlanningAgent 拆解
  2. 将子任务分发至对应专职 Agent
  3. 追踪任务依赖图 (DAG)，管理并行/串行执行
  4. 聚合 Agent 产出，形成最终交付物
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .message import Message, MessageType, Task, TaskStatus, AgentRole
from .agent import BaseAgent, AgentStatus
from .config import Config
from ..utils.logger import get_logger


class WorkflowStage(Enum):
    """工作流阶段"""
    INTAKE = "intake"              # 接收需求
    PLANNING = "planning"          # 规划拆解
    DISPATCHING = "dispatching"    # 任务分发
    EXECUTING = "executing"        # 并行执行
    INTEGRATING = "integrating"    # 结果整合
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowSession:
    """一次工作流会话"""
    session_id: str
    goal: str
    stage: WorkflowStage = WorkflowStage.INTAKE
    root_task: Optional[Task] = None
    task_results: dict[str, Task] = field(default_factory=dict)
    agent_outputs: dict[str, list[str]] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Orchestrator:
    """
    多 Agent 编排引擎

    工作流:
      INTAKE → PLANNING → DISPATCHING → EXECUTING → INTEGRATING → COMPLETED

    使用方式:
      orch = Orchestrator(config)
      orch.register_agent(PlanningAgent(...))
      orch.register_agent(CodingAgent(...))
      result = await orch.run("用 PyTorch 实现一个 VAE 模型")
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.logger = get_logger("Orchestrator")
        self.agents: dict[AgentRole, BaseAgent] = {}
        self.sessions: dict[str, WorkflowSession] = {}
        self._agent_tasks: dict[AgentRole, asyncio.Task] = {}

        os.makedirs(self.config.workspace_dir, exist_ok=True)

    def register_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.role] = agent
        self.logger.info(f"注册 Agent: {agent.role.value}")

    def unregister_agent(self, role: AgentRole) -> None:
        self.agents.pop(role, None)

    async def start(self) -> None:
        for agent in self.agents.values():
            await agent.start()
            self._agent_tasks[agent.role] = asyncio.create_task(agent.process_messages())
        self.logger.info(f"编排引擎启动，已注册 {len(self.agents)} 个 Agent")

    async def shutdown(self) -> None:
        for agent in self.agents.values():
            await agent.shutdown()
        for task in self._agent_tasks.values():
            task.cancel()
        self.logger.info("编排引擎已关闭")

    # ----------------------------------------------------------------
    # 主工作流
    # ----------------------------------------------------------------

    async def run(self, goal: str, context: dict | None = None) -> dict[str, Any]:
        """
        执行一个完整的多 Agent 工作流

        Args:
            goal: 用户目标描述
            context: 额外上下文（数据路径、参数等）

        Returns:
            包含各阶段产出与最终结果的字典
        """
        session = WorkflowSession(
            session_id=f"wf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            goal=goal,
        )
        self.sessions[session.session_id] = session

        try:
            # Stage 1: 需求规划
            session.stage = WorkflowStage.PLANNING
            self.logger.info(f"[{session.session_id}] 阶段: PLANNING")
            root_task = await self._plan(goal, context or {})
            session.root_task = root_task
            session.history.append({"stage": "planning", "task_tree": root_task.to_dict()})

            # Stage 2: 任务分发
            session.stage = WorkflowStage.DISPATCHING
            self.logger.info(f"[{session.session_id}] 阶段: DISPATCHING — {len(root_task.subtasks)} 个子任务")
            dispatch_plan = self._dispatch(root_task)

            # Stage 3: 并行执行
            session.stage = WorkflowStage.EXECUTING
            self.logger.info(f"[{session.session_id}] 阶段: EXECUTING")
            await self._execute_parallel(dispatch_plan, session)

            # Stage 4: 结果整合
            session.stage = WorkflowStage.INTEGRATING
            self.logger.info(f"[{session.session_id}] 阶段: INTEGRATING")
            final_result = await self._integrate(session)

            session.stage = WorkflowStage.COMPLETED
            self.logger.info(f"[{session.session_id}] 工作流完成")

            return {
                "session_id": session.session_id,
                "status": "completed",
                "goal": goal,
                "task_tree": root_task.to_dict(),
                "results": final_result,
            }

        except Exception as e:
            session.stage = WorkflowStage.FAILED
            self.logger.error(f"[{session.session_id}] 工作流失败: {e}")
            return {
                "session_id": session.session_id,
                "status": "failed",
                "error": str(e),
                "partial_results": session.agent_outputs,
            }

    # ----------------------------------------------------------------
    # Stage 1: 规划
    # ----------------------------------------------------------------

    async def _plan(self, goal: str, context: dict) -> Task:
        planner = self.agents.get(AgentRole.PLANNING)
        if not planner:
            raise RuntimeError("未注册 PlanningAgent")

        plan_task = Task(
            title="需求分析与任务拆解",
            description=goal,
            context={"raw_goal": goal, **context},
        )
        plan_task = await planner.execute(plan_task)

        if plan_task.status != TaskStatus.COMPLETED or not plan_task.result:
            raise RuntimeError(f"规划失败: {plan_task.error}")

        root_task = self._parse_task_tree(plan_task.result)
        return root_task

    def _parse_task_tree(self, plan_json: str) -> Task:
        """将 PlanningAgent 输出的 JSON 解析为 Task 树"""
        try:
            data = json.loads(plan_json)
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            import re
            m = re.search(r"\{[\s\S]*\}", plan_json)
            if m:
                data = json.loads(m.group())
            else:
                root = Task(title="未解析任务", description=plan_json[:500])
                return root

        return self._dict_to_task(data)

    def _dict_to_task(self, data: dict) -> Task:
        task = Task(
            title=data.get("title", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            context=data.get("context", {}),
        )
        assigned = data.get("assigned_to")
        if assigned:
            try:
                task.assigned_to = AgentRole(assigned)
            except ValueError:
                task.tags.append(f"role:{assigned}")

        for sub in data.get("subtasks", []):
            task.add_subtask(self._dict_to_task(sub))

        return task

    # ----------------------------------------------------------------
    # Stage 2: 分发
    # ----------------------------------------------------------------

    def _dispatch(self, root_task: Task) -> list[tuple[AgentRole, Task]]:
        dispatch_plan: list[tuple[AgentRole, Task]] = []
        self._collect_leaf_tasks(root_task, dispatch_plan)
        for role, task in dispatch_plan:
            self.logger.debug(f"分发: {task.title} → {role.value}")
        return dispatch_plan

    def _collect_leaf_tasks(self, task: Task, plan: list[tuple[AgentRole, Task]]) -> None:
        if task.is_leaf():
            role = self._resolve_role(task)
            plan.append((role, task))
        else:
            for sub in task.subtasks:
                self._collect_leaf_tasks(sub, plan)

    def _resolve_role(self, task: Task) -> AgentRole:
        if task.assigned_to and task.assigned_to in self.agents:
            return task.assigned_to

        title_lower = task.title.lower()
        desc_lower = task.description.lower()
        combined = f"{title_lower} {desc_lower}"

        code_keywords = ["写代码", "模型", "训练", "pytorch", "tensorflow", "函数", "类", "架构",
                         "代码", "编程", "实现", "vae", "transformer", "moe", "调试", "debug"]
        data_keywords = ["数据", "清洗", "归一", "标准化", "归一化", "z-score", "log变换", "特征",
                         "降噪", "处理", "表型", "csv", "dataframe", "pandas", "预处理"]
        lit_keywords = ["论文", "文献", "综述", "摘要", "提取", "精读", "框架", "参考"]
        ops_keywords = ["部署", "docker", "服务器", "环境", "配置", "linux", "容器", "后台",
                        "持久化", "远程"]

        if any(kw in combined for kw in code_keywords):
            return AgentRole.CODING
        elif any(kw in combined for kw in data_keywords):
            return AgentRole.DATA
        elif any(kw in combined for kw in lit_keywords):
            return AgentRole.LITERATURE
        elif any(kw in combined for kw in ops_keywords):
            return AgentRole.OPS

        return AgentRole.CODING  # 默认分配给代码 Agent

    # ----------------------------------------------------------------
    # Stage 3: 并行执行
    # ----------------------------------------------------------------

    async def _execute_parallel(
        self, dispatch_plan: list[tuple[AgentRole, Task]], session: WorkflowSession
    ) -> None:
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def bounded_execute(role: AgentRole, task: Task) -> None:
            async with semaphore:
                agent = self.agents.get(role)
                if not agent:
                    task.status = TaskStatus.CANCELLED
                    task.error = f"Agent {role.value} 未注册"
                    return

                try:
                    result_task = await agent.execute(task)
                    session.task_results[task.id] = result_task
                    if role.value not in session.agent_outputs:
                        session.agent_outputs[role.value] = []
                    session.agent_outputs[role.value].append(result_task.result or "")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

        coros = [bounded_execute(role, task) for role, task in dispatch_plan]
        await asyncio.gather(*coros, return_exceptions=True)

    # ----------------------------------------------------------------
    # Stage 4: 整合
    # ----------------------------------------------------------------

    async def _integrate(self, session: WorkflowSession) -> dict[str, Any]:
        completed = sum(1 for t in session.task_results.values()
                        if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in session.task_results.values()
                     if t.status == TaskStatus.FAILED)

        integration = {
            "summary": f"完成 {completed} 个子任务, 失败 {failed} 个",
            "by_agent": {},
        }

        for role_value, outputs in session.agent_outputs.items():
            integration["by_agent"][role_value] = {
                "output_count": len(outputs),
                "outputs": outputs,
            }

        return integration
