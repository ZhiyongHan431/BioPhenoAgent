"""
BaseAgent — 所有 Agent 的抽象基类，定义统一的接口与生命周期
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional

from .message import Message, MessageType, Task, TaskStatus, AgentRole
from .config import AgentConfig
from ..utils.logger import get_logger


class AgentStatus(Enum):
    """Agent 生命周期状态"""
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    TERMINATED = auto()


@dataclass
class AgentContext:
    """Agent 运行时上下文"""
    session_id: str = ""
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    tools_registry: dict[str, Callable] = field(default_factory=dict)


class BaseAgent(abc.ABC):
    """
    所有 Agent 的抽象基类

    子类必须实现:
      - _build_system_prompt() -> str: 返回 Agent 的系统提示词
      - _process_task(task: Task) -> str: 处理任务并返回结果
    """

    def __init__(
        self,
        role: AgentRole,
        config: AgentConfig | None = None,
        llm_client: Any = None,
    ):
        self.role = role
        self.config = config or AgentConfig(role=role.value)
        self.llm_client = llm_client
        self.status = AgentStatus.IDLE
        self.context = AgentContext()
        self.logger = get_logger(f"Agent.{role.value}")
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._handlers: dict[MessageType, Callable] = {
            MessageType.TASK_ASSIGNMENT: self._handle_task,
            MessageType.QUERY: self._handle_query,
            MessageType.COORDINATION: self._handle_coordination,
            MessageType.SYSTEM: self._handle_system,
        }

    # ---- 生命周期 ----

    async def start(self) -> None:
        self.status = AgentStatus.IDLE
        self.logger.info(f"[{self.role.value}] Agent 启动就绪")

    async def shutdown(self) -> None:
        self.status = AgentStatus.TERMINATED
        self.logger.info(f"[{self.role.value}] Agent 已关闭")

    # ---- 消息处理 ----

    async def receive(self, msg: Message) -> None:
        await self._message_queue.put(msg)

    async def process_messages(self) -> None:
        while self.status != AgentStatus.TERMINATED:
            try:
                msg = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                handler = self._handlers.get(msg.msg_type, self._default_handler)
                await handler(msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"[{self.role.value}] 消息处理异常: {e}")

    async def send(self, msg: Message, target_queue: Callable[[Message], Any]) -> None:
        msg.sender = self.role
        await target_queue(msg)

    # ---- 任务处理 ----

    async def execute(self, task: Task) -> Task:
        self.status = AgentStatus.BUSY
        task.status = TaskStatus.IN_PROGRESS
        self.logger.info(f"[{self.role.value}] 开始任务: {task.title}")

        try:
            task.result = await asyncio.wait_for(
                self._process_task_wrapper(task),
                timeout=self.config.timeout_seconds,
            )
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            self.logger.info(f"[{self.role.value}] 任务完成: {task.title}")
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"任务超时 ({self.config.timeout_seconds}s)"
            self.logger.error(f"[{self.role.value}] 任务超时: {task.title}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.logger.error(f"[{self.role.value}] 任务失败: {task.title} — {e}")
        finally:
            self.status = AgentStatus.IDLE

        return task

    async def _process_task_wrapper(self, task: Task) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(task)

        if self.llm_client:
            return await self._call_llm(system_prompt, user_prompt, task.context)
        else:
            return await self._process_task(task)

    @abc.abstractmethod
    async def _process_task(self, task: Task) -> str:
        """子类实现核心任务处理逻辑（无 LLM 时的回退方案）"""

    @abc.abstractmethod
    def _build_system_prompt(self) -> str:
        """构建 Agent 专用的系统提示词"""

    def _build_user_prompt(self, task: Task) -> str:
        return f"任务: {task.title}\n描述: {task.description}\n\n请完成此任务。"

    async def _call_llm(self, system_prompt: str, user_prompt: str, context: dict) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temperature = self.config.temperature_override or self.llm_client.temperature if hasattr(self.llm_client, 'temperature') else 0.3
        return await self.llm_client.generate(messages, temperature=temperature)

    # ---- 内部消息处理 ----

    async def _handle_task(self, msg: Message) -> None:
        task = Task(**msg.payload) if "id" in msg.payload else Task(description=msg.content)
        result = await self.execute(task)
        # 响应消息通过 payload 返回（由 Orchestrator 收集）

    async def _handle_query(self, msg: Message) -> None:
        pass

    async def _handle_coordination(self, msg: Message) -> None:
        pass

    async def _handle_system(self, msg: Message) -> None:
        if msg.content == "shutdown":
            await self.shutdown()

    async def _default_handler(self, msg: Message) -> None:
        self.logger.debug(f"[{self.role.value}] 未处理的消息类型: {msg.msg_type.name}")

    # ---- 工具注册 ----

    def register_tool(self, name: str, fn: Callable) -> None:
        self.context.tools_registry[name] = fn
