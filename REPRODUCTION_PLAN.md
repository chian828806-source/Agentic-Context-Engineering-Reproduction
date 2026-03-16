# ACE (Agentic Context Engineering) 复现方案

## 项目概述

本方案旨在完整复现ACE论文中的核心框架，在GLM-4.6上实现自进化的System Prompt优化系统。

---

## 一、ACE框架核心原理

### 1.1 三节点循环架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ACE进化循环                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────┐  │
│   │  Generator  │─────▶│  Reflector  │─────▶│ Curator │  │
│   │  (执行器)   │      │  (反思器)   │      │(策展器) │  │
│   └──────┬──────┘      └──────┬──────┘      └────┬────┘  │
│          │                    │                   │       │
│          ▼                    │                   │       │
│    ┌──────────┐               │                   │       │
│    │ Playbook │◀──────────────┘                   │       │
│    │ (上下文) │◀──────────────────────────────────┘       │
│    └──────────┘                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 三大组件职责

| 组件 | 输入 | 输出 | 功能 |
|:---|:---|:---|:---|
| **Generator** | 问题 + Playbook | 答案/代码 | 使用当前策略解决任务 |
| **Reflector** | 生成结果 + Ground Truth | 反思报告 | 分析差距，诊断错误原因 |
| **Curator** | Playbook + 反思报告 | 增量更新 | 精炼并扩展Playbook |

### 1.3 解决的核心问题

- **Brevity Bias (简洁性偏误)**: 传统方法倾向于压缩上下文，丢失领域洞察
- **Context Collapse (上下文坍塌)**: 迭代重写逐渐侵蚀细节信息
- **ACE解决方案**: 结构化的增量更新，保留并组织详细知识

---

## 二、项目架构设计

### 2.1 目录结构

```
ACE/
├── data/                           # 数据集目录
│   ├── gsm8k_train.jsonl          # GSM8K训练集
│   ├── gsm8k_test.jsonl           # GSM8K测试集
│   └── finicial/                   # 金融数据（可选）
│
├── src/
│   ├── nodes/                      # 核心节点
│   │   ├── __init__.py
│   │   ├── generator.py           # Generator节点
│   │   ├── reflector.py           # Reflector节点
│   │   ├── curator.py             # Curator节点
│   │   └── evaluator.py           # 评估节点
│   │
│   ├── state/                      # 状态管理
│   │   ├── __init__.py
│   │   └── graph_state.py         # LangGraph状态定义
│   │
│   ├── prompts/                    # 提示词模板
│   │   ├── __init__.py
│   │   ├── generator_prompts.py   # Generator提示词
│   │   ├── reflector_prompts.py   # Reflector提示词
│   │   └── curator_prompts.py     # Curator提示词
│   │
│   ├── llm/                        # LLM接口
│   │   ├── __init__.py
│   │   └── glm_client.py          # GLM-4.6 API客户端
│   │
│   ├── graph/                      # 流程图
│   │   ├── __init__.py
│   │   └── ace_graph.py           # 主流程图构建
│   │
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── data_loader.py         # 数据加载
│       ├── playbook.py            # Playbook数据结构
│       └── logger.py              # 日志记录
│
├── baselines/                      # 基线方法
│   ├── rag_baseline.py            # RAG基线
│   ├── icl_baseline.py            # In-Context Learning基线
│   └── fewshot_baseline.py        # Few-shot基线
│
├── logs/                           # 实验日志
│   ├── prompts_v{n}.json          # 每代演化的Prompt
│   ├── metrics.json               # 性能指标
│   └── evolution_history.json     # 演化历史
│
├── configs/                        # 配置文件
│   └── ace_config.yaml            # ACE配置
│
├── main.py                         # 主入口
├── requirements.txt                # 依赖
└── README.md
```

### 2.2 核心数据结构

```python
# src/state/graph_state.py
from typing import TypedDict, List, Dict, Any, Optional

class ACEState(TypedDict):
    """LangGraph状态定义"""
    # 当前Playbook
    current_playbook: Dict[str, List[str]]

    # 演化状态
    generation_index: int          # 当前代数
    fitness_score: float           # 当前适应度（准确率）
    error_samples: List[Dict]      # 错误样本（top 3-5）

    # 样本数据
    current_sample: Dict[str, Any] # 当前处理的样本
    ground_truth: Any              # Ground truth答案

    # Generator输出
    generated_answer: Any          # Generator生成的答案
    generator_trace: str           # Generator推理轨迹

    # Reflector输出
    reflection: Dict[str, Any]     # Reflector的反思报告

    # 配置
    max_generations: int           # 最大演化代数
    plateau_threshold: int         # 停滞阈值（连续N代无改善则停止）

    # 历史记录
    best_playbook: Dict[str, List[str]]  # 历史最佳Playbook
    best_score: float              # 历史最佳分数
    no_improvement_count: int      # 连续无改善次数
```

```python
# src/utils/playbook.py
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class Playbook:
    """ACE Playbook数据结构"""
    # 策略类别
    strategies_and_hard_rules: List[str] = field(default_factory=list)
    formulas_and_calculations: List[str] = field(default_factory=list)
    verification_checklist: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    apis_to_use_for_specific_information: List[str] = field(default_factory=list)

    # 元数据
    bullet_counter: Dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "strategies_and_hard_rules": self.strategies_and_hard_rules,
            "formulas_and_calculations": self.formulas_and_calculations,
            "verification_checklist": self.verification_checklist,
            "common_mistakes": self.common_mistakes,
            "apis_to_use_for_specific_information": self.apis_to_use_for_specific_information,
            "metadata": {
                "bullet_counter": self.bullet_counter,
                "created_at": self.created_at,
                "last_updated": self.last_updated
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Playbook':
        """从字典恢复"""
        metadata = data.get("metadata", {})
        return cls(
            strategies_and_hard_rules=data.get("strategies_and_hard_rules", []),
            formulas_and_calculations=data.get("formulas_and_calculations", []),
            verification_checklist=data.get("verification_checklist", []),
            common_mistakes=data.get("common_mistakes", []),
            apis_to_use_for_specific_information=data.get("apis_to_use_for_specific_information", []),
            bullet_counter=metadata.get("bullet_counter", {}),
            created_at=metadata.get("created_at", ""),
            last_updated=metadata.get("last_updated", "")
        )
```

---

## 三、核心实现细节

### 3.1 Generator节点实现

```python
# src/nodes/generator.py
from typing import Dict, Any
from ..llm.glm_client import GLMClient
from ..prompts.generator_prompts import get_generator_prompt

class GeneratorNode:
    """Generator节点：使用当前Playbook生成答案"""

    def __init__(self, llm_client: GLMClient, max_retries: int = 3):
        self.llm_client = llm_client
        self.max_retries = max_retries

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Generator节点

        Args:
            state: 包含current_playbook, current_sample的状态

        Returns:
            更新后的state，包含generated_answer和generator_trace
        """
        playbook = state["current_playbook"]
        sample = state["current_sample"]

        # 构建提示词（注意：仅发送必要的上下文）
        prompt = get_generator_prompt(
            question=sample["question"],
            playbook=playbook
        )

        # 带重试的LLM调用
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.call(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                continue

        # 解析响应（JSON格式）
        import json
        try:
            result = json.loads(response)
            return {
                "generated_answer": result.get("final_answer"),
                "generator_trace": result.get("reasoning", "")
            }
        except json.JSONDecodeError:
            # 回退：如果JSON解析失败，提取最后一行作为答案
            lines = response.strip().split('\n')
            return {
                "generated_answer": lines[-1] if lines else "",
                "generator_trace": response
            }
```

### 3.2 Reflector节点实现

```python
# src/nodes/reflector.py
from typing import Dict, Any
from ..llm.glm_client import GLMClient
from ..prompts.reflector_prompts import get_reflector_prompt

class ReflectorNode:
    """Reflector节点：分析错误并生成反思"""

    def __init__(self, llm_client: GLMClient, max_retries: int = 3):
        self.llm_client = llm_client
        self.max_retries = max_retries

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Reflector节点

        Args:
            state: 包含generated_answer, ground_truth, current_playbook等

        Returns:
            更新后的state，包含reflection
        """
        # 关键：仅发送必要信息，不发送完整聊天历史
        prompt = get_reflector_prompt(
            question=state["current_sample"]["question"],
            model_reasoning=state["generator_trace"],
            model_answer=state["generated_answer"],
            ground_truth=state["ground_truth"],
            playbook=state["current_playbook"]
        )

        # 带重试的LLM调用
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.call(
                    prompt=prompt,
                    temperature=0.3,  # 更低温度以获得更一致的反思
                    max_tokens=1500
                )
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                continue

        # 解析JSON响应
        import json
        reflection = json.loads(response)

        return {"reflection": reflection}
```

### 3.3 Curator节点实现

```python
# src/nodes/curator.py
from typing import Dict, Any
from ..llm.glm_client import GLMClient
from ..prompts.curator_prompts import get_curator_prompt
from ..utils.playbook import Playbook

class CuratorNode:
    """Curator节点：根据反思更新Playbook"""

    def __init__(self, llm_client: GLMClient, max_playbook_size: int = 1000):
        self.llm_client = llm_client
        self.max_playbook_size = max_playbook_size  # Token限制

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Curator节点（增量更新）

        Args:
            state: 包含current_playbook, reflection等

        Returns:
            更新后的state，包含更新后的current_playbook
        """
        playbook_dict = state["current_playbook"]
        reflection = state["reflection"]
        question = state["current_sample"]["question"]

        # 计算当前playbook大小
        current_size = len(str(playbook_dict))

        # 如果playbook太大，发送精简版给LLM
        if current_size > self.max_playbook_size * 0.8:
            playbook_dict = self._compress_playbook(playbook_dict)

        prompt = get_curator_prompt(
            question_context=question,
            current_playbook=playbook_dict,
            reflection=reflection,
            token_budget=self.max_playbook_size
        )

        response = self.llm_client.call(
            prompt=prompt,
            temperature=0.2,  # 低温度以获得一致的更新
            max_tokens=2000
        )

        # 解析并应用操作
        import json
        curator_result = json.loads(response)
        updated_playbook = self._apply_operations(
            playbook_dict,
            curator_result.get("operations", [])
        )

        return {"current_playbook": updated_playbook}

    def _compress_playbook(self, playbook: Dict) -> Dict:
        """压缩playbook以适应token限制"""
        # 简单策略：保留每个section的最新N条
        compressed = {}
        keep_per_section = 5
        for key, value in playbook.items():
            if isinstance(value, list) and len(value) > keep_per_section:
                compressed[key] = value[-keep_per_section:]
            else:
                compressed[key] = value
        return compressed

    def _apply_operations(self, playbook: Dict, operations: list) -> Dict:
        """应用Curator的操作到Playbook"""
        result = playbook.copy()

        for op in operations:
            op_type = op.get("type")
            section = op.get("section")
            content = op.get("content", "")

            if op_type == "ADD":
                if section not in result:
                    result[section] = []
                result[section].append(content)

        return result
```

### 3.4 Evaluator节点实现

```python
# src/nodes/evaluator.py
from typing import Dict, Any, List
import re

class EvaluatorNode:
    """Evaluator节点：评估当前Playbook的性能"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        在测试集上评估当前Playbook

        Returns:
            更新后的state，包含fitness_score和error_samples
        """
        # 这部分在主循环中批量处理
        # 这里仅定义接口
        pass

    @staticmethod
    def extract_numeric_answer(text: str) -> float:
        """从文本中提取数值答案"""
        # 尝试直接解析
        try:
            return float(text.strip())
        except ValueError:
            pass

        # 尝试从文本中提取数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[-1])  # 通常最后一个数字是答案
        return None

    @staticmethod
    def compare_answers(predicted: Any, ground_truth: Any) -> bool:
        """比较预测答案与ground truth"""
        pred_num = EvaluatorNode.extract_numeric_answer(str(predicted))
        gt_num = EvaluatorNode.extract_numeric_answer(str(ground_truth))

        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) < 1e-6

        return str(predicted).strip().lower() == str(ground_truth).strip().lower()
```

---

## 四、LangGraph流程构建

### 4.1 主循环图结构

```python
# src/graph/ace_graph.py
from typing import Literal
from langgraph.graph import StateGraph, END
from .state import ACEState
from ..nodes.generator import GeneratorNode
from ..nodes.reflector import ReflectorNode
from ..nodes.curator import CuratorNode
from ..nodes.evaluator import EvaluatorNode

class ACEGraph:
    """ACE主流程图"""

    def __init__(self, llm_client, config: Dict):
        self.llm_client = llm_client
        self.config = config

        # 初始化节点
        self.generator = GeneratorNode(llm_client)
        self.reflector = ReflectorNode(llm_client)
        self.curator = CuratorNode(llm_client)
        self.evaluator = EvaluatorNode(llm_client)

    def build(self) -> StateGraph:
        """构建ACE流程图"""
        graph = StateGraph(ACEState)

        # 添加节点
        graph.add_node("generator", self._generator_node)
        graph.add_node("reflector", self._reflector_node)
        graph.add_node("curator", self._curator_node)
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("check_convergence", self._check_convergence_node)

        # 设置入口
        graph.set_entry_point("generator")

        # 添加边
        graph.add_edge("generator", "reflector")
        graph.add_edge("reflector", "curator")
        graph.add_edge("curator", "evaluator")
        graph.add_edge("evaluator", "check_convergence")

        # 条件边：是否继续演化
        graph.add_conditional_edges(
            "check_convergence",
            self._should_continue,
            {
                "continue": "generator",   # 继续下一轮
                "end": END                 # 结束
            }
        )

        return graph.compile()

    def _generator_node(self, state: ACEState) -> ACEState:
        return {**state, **self.generator(state)}

    def _reflector_node(self, state: ACEState) -> ACEState:
        return {**state, **self.reflector(state)}

    def _curator_node(self, state: ACEState) -> ACEState:
        return {**state, **self.curator(state)}

    def _evaluator_node(self, state: ACEState) -> ACEState:
        # 在实际实现中，这里会在验证集上批量评估
        # 简化版本：更新fitness_score
        return {**state, **self.evaluator(state)}

    def _check_convergence_node(self, state: ACEState) -> ACEState:
        """检查是否收敛"""
        # 更新最佳记录
        if state["fitness_score"] > state["best_score"]:
            state["best_score"] = state["fitness_score"]
            state["best_playbook"] = state["current_playbook"].copy()
            state["no_improvement_count"] = 0
        else:
            state["no_improvement_count"] += 1

        return state

    def _should_continue(self, state: ACEState) -> Literal["continue", "end"]:
        """判断是否继续演化"""
        # 停止条件：
        # 1. 达到最大代数
        if state["generation_index"] >= state["max_generations"]:
            return "end"

        # 2. 连续N代无改善
        if state["no_improvement_count"] >= state["plateau_threshold"]:
            return "end"

        return "continue"
```

### 4.2 训练循环实现

```python
# main.py (核心训练逻辑)
import json
from pathlib import Path
from src.graph.ace_graph import ACEGraph
from src.llm.glm_client import GLMClient
from src.utils.data_loader import load_gsm8k
from src.utils.playbook import Playbook
from src.state.graph_state import ACEState

def initialize_playbook() -> dict:
    """初始化空Playbook或基础Prompt"""
    return {
        "strategies_and_hard_rules": [
            "Carefully read the problem and identify what is being asked.",
            "Break down complex problems into smaller, manageable steps.",
            "Show your work step by step for clarity."
        ],
        "formulas_and_calculations": [],
        "verification_checklist": [
            "Double-check your calculations before finalizing the answer.",
            "Ensure the answer matches the units requested in the question."
        ],
        "common_mistakes": [],
        "apis_to_use_for_specific_information": []
    }

def train_ace(config: dict):
    """主训练循环"""
    # 初始化
    llm_client = GLMClient(
        api_key=config["api_key"],
        base_url=config.get("base_url", "https://open.bigmodel.cn/api/paas/v4/"),
        model=config.get("model", "glm-4.6")
    )

    # 加载数据
    train_data = load_gsm8k(config["train_path"])
    test_data = load_gsm8k(config["test_path"])

    # 初始化状态
    initial_state: ACEState = {
        "current_playbook": initialize_playbook(),
        "generation_index": 0,
        "fitness_score": 0.0,
        "error_samples": [],
        "current_sample": {},
        "ground_truth": None,
        "generated_answer": None,
        "generator_trace": "",
        "reflection": {},
        "max_generations": config.get("max_generations", 10),
        "plateau_threshold": config.get("plateau_threshold", 3),
        "best_playbook": initialize_playbook(),
        "best_score": 0.0,
        "no_improvement_count": 0
    }

    # 构建图
    ace_graph = ACEGraph(llm_client, config).build()

    # 训练循环（离线多轮）
    logs_dir = Path(config["logs_dir"])
    logs_dir.mkdir(exist_ok=True)

    for epoch in range(config.get("num_epochs", 5)):
        print(f"\n=== Epoch {epoch + 1} ===")

        for sample_idx, sample in enumerate(train_data):
            # 更新当前样本
            initial_state["current_sample"] = sample
            initial_state["ground_truth"] = sample["answer"]

            # 执行ACE循环
            result = ace_graph.invoke(initial_state)

            # 更新状态
            initial_state["current_playbook"] = result["current_playbook"]
            initial_state["generation_index"] = result["generation_index"]
            initial_state["fitness_score"] = result["fitness_score"]

            # 定期评估
            if (sample_idx + 1) % config.get("eval_interval", 50) == 0:
                accuracy = evaluate_on_test_set(
                    llm_client,
                    result["current_playbook"],
                    test_data[:100]  # 使用100个样本快速评估
                )
                print(f"Sample {sample_idx + 1}, Test Accuracy: {accuracy:.2%}")

                # 保存当前最佳Playbook
                if accuracy > initial_state["best_score"]:
                    initial_state["best_score"] = accuracy
                    initial_state["best_playbook"] = result["current_playbook"].copy()

                    # 保存到文件
                    save_path = logs_dir / f"prompts_v{result['generation_index']}.json"
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "generation": result["generation_index"],
                            "accuracy": accuracy,
                            "playbook": result["current_playbook"]
                        }, f, indent=2, ensure_ascii=False)

    print("\n=== Training Complete ===")
    print(f"Best Accuracy: {initial_state['best_score']:.2%}")

def evaluate_on_test_set(llm_client, playbook, test_samples):
    """在测试集上评估"""
    from src.nodes.generator import GeneratorNode
    from src.nodes.evaluator import EvaluatorNode

    generator = GeneratorNode(llm_client)
    correct = 0

    for sample in test_samples:
        state = {
            "current_playbook": playbook,
            "current_sample": sample
        }
        result = generator(state)

        if EvaluatorNode.compare_answers(
            result["generated_answer"],
            sample["answer"]
        ):
            correct += 1

    return correct / len(test_samples)

if __name__ == "__main__":
    config = {
        "api_key": "your_api_key_here",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-4.6",
        "train_path": "data/gsm8k_train.jsonl",
        "test_path": "data/gsm8k_test.jsonl",
        "logs_dir": "logs",
        "max_generations": 10,
        "plateau_threshold": 3,
        "num_epochs": 5,
        "eval_interval": 50
    }

    train_ace(config)
```

---

## 五、提示词设计

### 5.1 Generator提示词（GSM8K版）

```python
# src/prompts/generator_prompts.py

GENERATOR_PROMPT_TEMPLATE = """You are an expert mathematical problem solver. Your job is to solve math word problems step by step.

You are provided with a curated playbook of strategies, formulas, and insights to help you solve problems effectively.

PLAYBOOK:
{playbook}

INSTRUCTIONS:
- Read the playbook carefully and apply relevant strategies and formulas
- Pay attention to common mistakes listed in the playbook and avoid them
- Show your reasoning step by step
- Be concise but thorough in your analysis
- Double-check your calculations before providing the final answer

QUESTION:
{question}

Your output should be a JSON object with these exact fields:
- reasoning: your step-by-step thinking process
- final_answer: your concise final answer (just the numeric value)

Answer in this exact JSON format:
{{
  "reasoning": "[Your step-by-step reasoning here]",
  "final_answer": "[Your final numeric answer]"
}}"""

def get_generator_prompt(question: str, playbook: dict) -> str:
    """构建Generator提示词"""
    # 格式化playbook为文本
    playbook_text = format_playbook(playbook)

    return GENERATOR_PROMPT_TEMPLATE.format(
        playbook=playbook_text,
        question=question
    )

def format_playbook(playbook: dict) -> str:
    """将playbook格式化为可读文本"""
    sections = []
    for section_name, items in playbook.items():
        if items:
            sections.append(f"\n## {section_name.replace('_', ' ').title()}")
            for i, item in enumerate(items, 1):
                sections.append(f"{i}. {item}")

    return "\n".join(sections) if sections else "No playbook entries yet."
```

### 5.2 Reflector提示词（GSM8K版）

```python
# src/prompts/reflector_prompts.py

REFLECTOR_PROMPT_TEMPLATE = """You are an expert math educator. Your job is to analyze why a solution went wrong by comparing the predicted answer with the ground truth.

INSTRUCTIONS:
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Compare the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Focus on the root cause, not just surface-level errors
- Be specific about what the model should have done differently

QUESTION:
{question}

MODEL'S REASONING:
{model_reasoning}

MODEL'S ANSWER:
{model_answer}

GROUND TRUTH ANSWER:
{ground_truth}

CURRENT PLAYBOOK:
{playbook}

Your output should be a JSON object with these exact fields:
- reasoning: your analysis of what went wrong
- error_identification: what specifically went wrong in the reasoning?
- root_cause_analysis: why did this error occur? What concept was misunderstood?
- correct_approach: what should the model have done instead?
- key_insight: what strategy, formula, or principle should be remembered to avoid this error?

Answer in this exact JSON format:
{{
  "reasoning": "[Your analysis here]",
  "error_identification": "[What went wrong]",
  "root_cause_analysis": "[Why it occurred]",
  "correct_approach": "[What should have been done]",
  "key_insight": "[Key principle to remember]"
}}"""

def get_reflector_prompt(question: str, model_reasoning: str,
                         model_answer: str, ground_truth: str,
                         playbook: dict) -> str:
    """构建Reflector提示词"""
    playbook_text = format_playbook(playbook)

    return REFLECTOR_PROMPT_TEMPLATE.format(
        question=question,
        model_reasoning=model_reasoning,
        model_answer=model_answer,
        ground_truth=ground_truth,
        playbook=playbook_text
    )
```

### 5.3 Curator提示词（GSM8K版）

```python
# src/prompts/curator_prompts.py

CURATOR_PROMPT_TEMPLATE = """You are a master curator of mathematical knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

CONTEXT:
- The playbook you created will be used to help solve similar math problems.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used.

CRITICAL INSTRUCTIONS:
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused playbook is better than an exhaustive one
- Output ONLY a valid JSON object (no markdown, no code blocks)

TRAINING CONTEXT:
- Total token budget: {token_budget} tokens
- Training progress: Sample {current_step} out of {total_samples}

CURRENT PLAYBOOK:
{current_playbook}

RECENT REFLECTION:
{recent_reflection}

QUESTION CONTEXT:
{question_context}

Your output should be a JSON object with these exact fields:
- reasoning: your chain of thought
- operations: a list of operations to perform on the playbook

Available operations:
1. ADD: Create new bullet points
   - section: the section to add to (strategies_and_hard_rules, formulas_and_calculations, verification_checklist, common_mistakes)
   - content: the new content

RESPONSE FORMAT (JSON only, no markdown):
{{
  "reasoning": "[Your reasoning here]",
  "operations": [
    {{
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "[New strategy or rule]"
    }}
  ]
}}"""

def get_curator_prompt(question_context: str, current_playbook: dict,
                       reflection: dict, token_budget: int,
                       current_step: int, total_samples: int) -> str:
    """构建Curator提示词"""
    playbook_text = format_playbook(current_playbook)

    # 格式化reflection
    reflection_text = f"""
Error Identification: {reflection.get('error_identification', '')}
Root Cause: {reflection.get('root_cause_analysis', '')}
Correct Approach: {reflection.get('correct_approach', '')}
Key Insight: {reflection.get('key_insight', '')}
"""

    return CURATOR_PROMPT_TEMPLATE.format(
        token_budget=token_budget,
        current_step=current_step,
        total_samples=total_samples,
        current_playbook=playbook_text,
        recent_reflection=reflection_text.strip(),
        question_context=question_context
    )
```

---

## 六、GLM-4.6 API集成

### 6.1 GLM客户端实现

```python
# src/llm/glm_client.py
import os
import time
from typing import Optional, Dict, Any
from openai import OpenAI

class GLMClient:
    """GLM-4.6 API客户端 (OpenAI兼容接口)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        model: str = "glm-4.6",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        调用GLM-4.6 API

        Args:
            prompt: 输入提示词
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式 (如 {"type": "json_object"})

        Returns:
            模型响应文本
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise RuntimeError(f"GLM API call failed after {self.max_retries} attempts: {e}")

    def call_with_messages(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        使用消息列表调用API

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            模型响应文本
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise RuntimeError(f"GLM API call failed: {e}")
```

---

## 七、基线方法实现

### 7.1 RAG基线

```python
# baselines/rag_baseline.py
"""
RAG基线：使用检索增强生成
"""
import json
from typing import List, Dict
from pathlib import Path

class RAGBaseline:
    """简单的RAG基线实现"""

    def __init__(self, llm_client, examples_path: str):
        self.llm_client = llm_client
        self.examples = self._load_examples(examples_path)

    def _load_examples(self, path: str) -> List[Dict]:
        """加载示例"""
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _retrieve_examples(self, question: str, k: int = 5) -> List[Dict]:
        """检索相关示例（简化版：基于关键词匹配）"""
        question_words = set(question.lower().split())

        scored_examples = []
        for ex in self.examples:
            ex_words = set(ex.get("question", "").lower().split())
            overlap = len(question_words & ex_words)
            scored_examples.append((overlap, ex))

        # 返回top-k
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:k]]

    def solve(self, question: str) -> str:
        """使用RAG解决问题"""
        examples = self._retrieve_examples(question)

        # 构建提示词
        prompt = "You are a math problem solver. Here are some similar examples:\n\n"
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

        prompt += f"Now solve this:\nQuestion: {question}\n"
        prompt += "Give your final answer as a number."

        response = self.llm_client.call(prompt, temperature=0.3)
        return response

    def evaluate(self, test_data: List[Dict]) -> float:
        """评估RAG基线"""
        correct = 0
        for sample in test_data:
            answer = self.solve(sample["question"])
            # 简化的答案比较
            if str(sample["answer"]) in answer or answer in str(sample["answer"]):
                correct += 1
        return correct / len(test_data)
```

### 7.2 Few-shot基线

```python
# baselines/fewshot_baseline.py
"""
Few-shot基线：固定示例的提示学习
"""
import json
from typing import List, Dict

class FewShotBaseline:
    """Few-shot基线"""

    # 固定的few-shot示例
    FEWSHOT_EXAMPLES = [
        {
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "72"
        },
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "answer": "10"
        },
        {
            "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
            "answer": "5"
        }
    ]

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def solve(self, question: str) -> str:
        """使用few-shot提示解决问题"""
        prompt = "Solve the following math problems step by step.\n\n"

        for i, ex in enumerate(self.FEWSHOT_EXAMPLES, 1):
            prompt += f"Q{i}: {ex['question']}\n"
            prompt += f"A{i}: {ex['answer']}\n\n"

        prompt += f"Q{len(self.FEWSHOT_EXAMPLES) + 1}: {question}\n"
        prompt += f"A{len(self.FEWSHOT_EXAMPLES) + 1}: "

        response = self.llm_client.call(prompt, temperature=0.0)
        return response.strip()

    def evaluate(self, test_data: List[Dict]) -> float:
        """评估Few-shot基线"""
        correct = 0
        for sample in test_data:
            answer = self.solve(sample["question"])
            if str(sample["answer"]) in answer:
                correct += 1
        return correct / len(test_data)
```

---

## 八、实验配置

### 8.1 配置文件

```yaml
# configs/ace_config.yaml
# ACE实验配置

# LLM配置
llm:
  model: "glm-4.6"
  base_url: "https://open.bigmodel.cn/api/paas/v4/"
  api_key_env: "ZHIPUAI_API_KEY"
  temperature:
    generator: 0.7
    reflector: 0.3
    curator: 0.2
  max_tokens:
    generator: 2048
    reflector: 1500
    curator: 2000

# 数据配置
data:
  train_path: "data/gsm8k_train.jsonl"
  test_path: "data/gsm8k_test.jsonl"
  train_size: 1000    # 使用1000个训练样本
  test_size: 100      # 使用100个测试样本

# 训练配置
training:
  max_generations: 10      # 每个样本最大演化代数
  num_epochs: 5            # 训练轮数
  plateau_threshold: 3     # 连续3代无改善则停止
  eval_interval: 50        # 每50个样本评估一次
  batch_size: 1            # 批量大小（论文中为1）

# Playbook配置
playbook:
  max_size: 1000           # 最大token数
  sections:
    - strategies_and_hard_rules
    - formulas_and_calculations
    - verification_checklist
    - common_mistakes
    - apis_to_use_for_specific_information

# 日志配置
logging:
  logs_dir: "logs"
  save_every_n_samples: 10
  verbose: true

# 评估配置
evaluation:
  metrics:
    - accuracy
    - token_usage
  compare_baselines:
    - rag
    - fewshot
```

---

## 九、实施步骤

### 阶段一：基础框架搭建（第1-2天）

1. **环境设置**
   - 创建项目目录结构
   - 安装依赖包
   - 配置GLM-4.6 API访问

2. **核心数据结构**
   - 实现`ACEState`
   - 实现`Playbook`类
   - 实现数据加载器

3. **LLM客户端**
   - 实现`GLMClient`
   - 添加重试逻辑
   - 测试API连接

### 阶段二：节点实现（第3-4天）

1. **Generator节点**
   - 实现Generator提示词模板
   - 实现Generator节点类
   - 单元测试

2. **Reflector节点**
   - 实现Reflector提示词模板
   - 实现Reflector节点类
   - 单元测试

3. **Curator节点**
   - 实现Curator提示词模板
   - 实现Curator节点类
   - 实现增量更新逻辑
   - 单元测试

### 阶段三：流程图构建（第5天）

1. **LangGraph集成**
   - 构建ACE流程图
   - 添加条件边逻辑
   - 实现收敛检查

2. **训练循环**
   - 实现主训练函数
   - 添加定期评估
   - 实现checkpoint保存

### 阶段四：基线方法（第6天）

1. **RAG基线**
   - 实现简单检索
   - 构建RAG提示词
   - 评估函数

2. **Few-shot基线**
   - 准备few-shot示例
   - 实现基线评估

### 阶段五：实验运行（第7天）

1. **小规模测试**
   - 使用10个样本测试
   - 验证各节点正常工作
   - 调试提示词

2. **完整实验**
   - 运行完整训练
   - 记录指标
   - 与基线比较

### 阶段六：分析与文档（第8天）

1. **结果分析**
   - 绘制学习曲线
   - 分析演化的Playbook
   - 比较与基线

2. **文档整理**
   - 更新README
   - 整理实验报告
   - 准备提交材料

---

## 十、依赖清单

```txt
# requirements.txt
# 核心依赖
openai>=1.0.0           # OpenAI兼容API客户端
langgraph>=0.0.0        # LangGraph框架
langchain>=0.1.0        # LangChain基础
pydantic>=2.0.0         # 数据验证

# 数据处理
datasets>=2.0.0         # Hugging Face数据集
pandas>=2.0.0           # 数据处理
numpy>=1.24.0           # 数值计算

# 工具库
python-dotenv>=1.0.0    # 环境变量
pyyaml>=6.0             # 配置文件
tqdm>=4.65.0            # 进度条
requests>=2.31.0        # HTTP请求

# 可选：日志和监控
wandb>=0.15.0           # 实验跟踪（可选）
tensorboard>=2.14.0     # 可视化（可选）

# 开发工具
pytest>=7.4.0           # 测试
black>=23.0.0           # 代码格式化
isort>=5.12.0           # Import排序
```

---

## 十一、注意事项

### 11.1 关键约束

1. **状态less优化**
   - Reflector/Curator调用时**不发送完整聊天历史**
   - 仅发送：当前prompt + top 3-5个失败案例 + 演化指令

2. **Token效率**
   - Worker(Gemerator)节点使用简洁提示词
   - 定期压缩Playbook避免token溢出

3. **回滚机制**
   - API调用失败时回滚到上一最佳prompt
   - 连续3代无改善时停止

### 11.2 论文对照检查

| 论文要素 | 实现位置 | 检查点 |
|:---|:---|:---|
| 三节点循环 | `src/graph/ace_graph.py` | Generator→Reflector→Curator |
| 离线适配 | `main.py` | 多epoch训练 |
| 在线适配 | 待扩展 | 实时样本处理 |
| 增量更新 | `src/nodes/curator.py` | 仅添加新内容 |
| 防坍塌机制 | `curator._compress_playbook()` | 定期压缩 |
| RAG基线 | `baselines/rag_baseline.py` | 检索增强 |

### 11.3 潜在挑战

1. **GLM-4.6 API稳定性**
   - 添加完善的重试机制
   - 考虑速率限制

2. **JSON解析可靠性**
   - 添加多个解析fallback
   - 考虑使用`response_format={"type": "json_object"}`

3. **评估指标准确性**
   - GSM8K答案提取
   - 数值比较容差

---

## 十二、成功标准

### 最小可行目标

- [ ] 完整实现三节点循环
- [ ] 在GSM8K上运行完成至少5代演化
- [ ] 比Base LLM有所提升（>1%）
- [ ] 实现RAG基线比较

### 理想目标

- [ ] 在GSM8K上达到论文报告的提升幅度（+5-10%）
- [ ] 完整的离线+在线两种模式
- [ ] 与RAG、Few-shot、ICL基线全面比较
- [ ] 可视化演化过程

---

## 十三、下一步行动

请确认以下事项后开始实施：

1. ✅ GLM-4.6 API密钥已获取
2. ✅ GSM8K数据集已下载/准备
3. ✅ Python 3.9+环境已配置
4. ⬜ 确认项目结构是否需要调整
5. ⬜ 确认优先级：是否先做简化版本验证

---

*本文档基于ACE论文 (ICLR 2026) 和项目CLAUDE.md要求生成*
