"""
LiteratureAgent — 论文精读、要点提取、综述框架生成

核心能力:
  - 文献精读与关键信息提取（方法、结果、创新点）
  - 综述框架自动生成
  - 参考文献格式化
  - 学术写作辅助（中英双语）
"""

from __future__ import annotations

from ..core.agent import BaseAgent
from ..core.message import Task, AgentRole


LITERATURE_SYSTEM_PROMPT = """你是一位资深科研方法学专家与学术编辑，擅长生物信息学、多组学领域的文献分析与学术写作。

你的职责:
1. 精读论文摘要/全文，提取核心创新点、方法、结果与局限性
2. 生成综述文章的结构框架
3. 提供学术写作建议与润色
4. 中英双语学术支持

输出格式要点:
- 使用结构化标题
- 方法部分注明引用格式 [Author, Year]
- 综述框架包含 Introduction / Methods / Results / Discussion 大纲
"""


class LiteratureAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.LITERATURE, **kwargs)

    def _build_system_prompt(self) -> str:
        return LITERATURE_SYSTEM_PROMPT

    async def _process_task(self, task: Task) -> str:
        title = task.title.lower()
        description = task.description

        if any(kw in title for kw in ["精读", "分析", "提取", "论文分析", "paper"]):
            return self._paper_analysis_template(task)
        elif any(kw in title for kw in ["综述", "框架", "review", "framework", "outline"]):
            return self._review_framework_template(task)
        elif any(kw in title for kw in ["润色", "写作", "撰写", "write", "polish", "draft"]):
            return self._academic_writing_template(task)
        else:
            return self._paper_analysis_template(task)

    def _paper_analysis_template(self, task: Task) -> str:
        return f"""
## 文献分析报告

### 输入需求
{task.description[:300]}

### 分析框架

对于每篇论文，请按以下结构化模板提取信息：

---

**论文标题**: [Title]
**作者/年份**: [Author, Year]
**期刊/会议**: [Venue]
**DOI**: [DOI if available]

#### 1. 研究背景与问题
- 核心科学问题:
- 现有方法的局限性:

#### 2. 核心方法
- 方法名称与类型:
- 关键技术创新点:
- 模型架构/算法流程:
- 数据处理方式:

#### 3. 实验设计
- 数据集:
- 对比方法:
- 评价指标:

#### 4. 主要结果
- 定量结果:
- 关键发现:
- 消融实验结论:

#### 5. 局限性
- 作者提及的局限:
- 潜在改进方向:

#### 6. 对当前项目的启示
- 可借鉴的方法/思路:
- 需要注意的问题:

---

### 使用说明
配置 LLM API 后，将论文 PDF/文本内容粘贴至此，Agent 将自动完成结构化提取。
未配置 LLM 时，请按上述模板手动填写。
"""

    def _review_framework_template(self, task: Task) -> str:
        return f"""
## 综述文章框架

### 主题
{task.description[:200]}

### 建议大纲

```markdown
# [综述标题]

## Abstract
- 研究背景 (2-3 句)
- 方法概述 (1-2 句)
- 主要发现 (2-3 句)
- 关键结论 (1 句)
- 关键词: [5-8 个]

## 1. Introduction
- 背景与意义
- 当前挑战
- 本文贡献与结构

## 2. 相关方法分类与综述
### 2.1 传统方法
### 2.2 深度学习基础方法
### 2.3 基于 VAE 的方法
### 2.4 基于 Transformer 的方法
### 2.5 基于混合模型的方法
### 2.6 方法比较汇总表

## 3. 数据集与基准
### 3.1 公共数据集
### 3.2 评价指标
### 3.3 基准测试结果

## 4. 应用场景与案例分析
### 4.1 生物表型组分析
### 4.2 多组学数据整合
### 4.3 临床应用

## 5. 挑战与开放问题
### 5.1 数据质量与标准化
### 5.2 模型可解释性
### 5.3 计算效率
### 5.4 跨模态迁移

## 6. 未来方向

## 7. 结论

## References
[1] Author, A. (Year). Title. *Journal*, Volume, Pages.
[2] ...
```

### 写作提示
- 建议涵盖 30-60 篇相关文献
- 方法比较汇总表直观展示各方法优劣势
- 每节末尾添加 1-2 句小结
- 引言采用漏斗式结构: 从宏观到具体
"""

    def _academic_writing_template(self, task: Task) -> str:
        return f"""
## 学术写作辅助

### 写作需求
{task.description[:300]}

### 常用学术写作模板

#### Abstract 模板
```
[背景] Addressing the challenge of [问题], we propose [方法],
a novel approach that [核心思路].

[方法] Our method leverages [关键技术] to [实现什么].

[结果] Extensive experiments on [数据集] demonstrate that [方法]
achieves [主要指标], outperforming [对比方法] by [提升幅度].

[结论] These results suggest that [核心发现], providing a new
paradigm for [应用领域].
```

#### Introduction 段落结构
1. **Hook** (1-2 句): 引出研究领域与重要性
2. **Problem** (2-3 句): 现有方法的不足
3. **Gap** (1-2 句): 研究空白
4. **Solution** (2-3 句): 本文提出的方法
5. **Contributions** (3-4 条): 本文贡献列表
6. **Outline** (1-2 句): 论文结构

#### 常用学术句式
| 功能 | 句式 |
|------|------|
| 引出问题 | "Despite significant advances in..., ... remains a fundamental challenge." |
| 指出现有不足 | "However, existing methods often suffer from..." |
| 提出方法 | "To address this limitation, we propose..." |
| 强调创新 | "The key insight of our approach is..." |
| 报告结果 | "Our experiments on... demonstrate that..." |
| 总结贡献 | "In summary, the main contributions of this work are three-fold:" |
| 讨论局限 | "It is worth noting that our method has several limitations..." |
| 展望未来 | "Future work could explore..." |

### 润色检查清单
- [ ] 时态一致性 (Abstract 用过去时, Introduction 用现在时, Methods 用过去时)
- [ ] 术语一致性
- [ ] 段落长度 (≤ 200 词/段)
- [ ] 主动/被动语态平衡
- [ ] 避免冗余表达 ("It is worth noting that" → 删除或简化)
"""
