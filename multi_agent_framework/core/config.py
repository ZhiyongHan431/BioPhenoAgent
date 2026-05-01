"""
配置管理 — LLM 后端、Agent 参数、系统级配置
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMBackend:
    """LLM 后端配置"""
    provider: str = "openai"  # openai | anthropic | local
    model: str = "gpt-4"
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    api_base: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 8192
    top_p: float = 0.95
    extra_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> LLMBackend:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentConfig:
    """单个 Agent 的配置"""
    role: str = ""
    system_prompt: str = ""
    max_retries: int = 3
    timeout_seconds: int = 300
    tools: list[str] = field(default_factory=list)
    temperature_override: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> AgentConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """系统全局配置"""
    llm: LLMBackend = field(default_factory=LLMBackend)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    workspace_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "workspace"))
    max_concurrent_tasks: int = 5
    log_level: str = "INFO"
    log_file: Optional[str] = "agent_framework.log"
    context_window_limit: int = 100000
    session_persist: bool = True

    @classmethod
    def from_file(cls, path: str) -> Config:
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        elif path.endswith((".yaml", ".yml")):
            try:
                import yaml
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("需要安装 PyYAML: pip install pyyaml")
        else:
            raise ValueError(f"不支持的配置文件格式: {path}")

        cfg = cls()
        if "llm" in data:
            cfg.llm = LLMBackend.from_dict(data["llm"])
        if "agents" in data:
            cfg.agents = {k: AgentConfig.from_dict(v) for k, v in data["agents"].items()}
        for key in ("workspace_dir", "max_concurrent_tasks", "log_level", "log_file",
                     "context_window_limit", "session_persist"):
            if key in data:
                setattr(cfg, key, data[key])
        return cfg

    def to_dict(self) -> dict:
        return {
            "llm": {k: v for k, v in self.llm.__dict__.items() if not k.startswith("_")},
            "agents": {k: {ak: av for ak, av in v.__dict__.items() if not ak.startswith("_")}
                       for k, v in self.agents.items()},
            "workspace_dir": self.workspace_dir,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }
