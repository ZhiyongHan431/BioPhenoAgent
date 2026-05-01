"""
消息系统 — Agent 间通信的标准化数据类型
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


class MessageType(Enum):
    """消息类型"""
    TASK_ASSIGNMENT = auto()    # 任务分配
    TASK_RESULT = auto()        # 任务结果
    STATUS_UPDATE = auto()      # 状态更新
    QUERY = auto()              # 查询请求
    RESPONSE = auto()           # 查询响应
    ERROR = auto()              # 错误通知
    COORDINATION = auto()       # Agent 间协调
    SYSTEM = auto()             # 系统级消息


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """Agent 角色标识"""
    PLANNING = "planning"
    CODING = "coding"
    DATA = "data"
    LITERATURE = "literature"
    OPS = "ops"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Message:
    """Agent 间通信的标准消息体"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    msg_type: MessageType = MessageType.SYSTEM
    sender: AgentRole = AgentRole.ORCHESTRATOR
    recipient: AgentRole = AgentRole.ORCHESTRATOR
    content: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None  # 关联消息 ID，用于追踪对话链
    priority: int = 0  # 0=普通, 1=高, 2=紧急

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "msg_type": self.msg_type.name,
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "content": self.content,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
        }


@dataclass
class Task:
    """可被分配、追踪的任务体"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    assigned_to: Optional[AgentRole] = None
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    subtasks: list[Task] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "assigned_to": self.assigned_to.value if self.assigned_to else None,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "result": self.result,
        }

    def add_subtask(self, task: Task) -> None:
        task.parent_id = self.id
        self.subtasks.append(task)

    def is_leaf(self) -> bool:
        return len(self.subtasks) == 0
