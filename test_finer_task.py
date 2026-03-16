"""
FINER任务测试脚本
用于验证FINER任务的实现是否正确
"""

import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.data_loader import load_finer
from src.prompts.generator_prompts import get_generator_prompt
from src.nodes.generator import GeneratorNode
from src.llm.glm_client import GLMClient
from src.utils.env import load_env

def test_finer_data_loading():
    """测试FINER数据加载"""
    print("=== 测试FINER数据加载 ===")

    try:
        # 加载环境变量
        load_env()

        # 加载FINER数据
        train_data = load_finer("data/train.jsonl", max_samples=5)
        val_data = load_finer("data/test.jsonl", max_samples=3)

        print(f"✅ FINER训练数据加载成功，样本数: {len(train_data)}")
        print(f"✅ FINER验证数据加载成功，样本数: {len(val_data)}")

        # 打印第一个样本
        if train_data:
            print(f"第一个训练样本: {train_data[0]}")

        return train_data, val_data
    except Exception as e:
        print(f"❌ FINER数据加载失败: {e}")
        return None, None

def test_finer_prompt_generation(train_data=None):
    """测试FINER提示词生成"""
    print("\n=== 测试FINER提示词生成 ===")

    try:
        # 加载环境变量
        load_env()

        # 使用训练数据中的样本，如果未提供则使用默认值
        if train_data and len(train_data) > 0:
            sample = train_data[0]  # 使用第一个训练样本
        else:
            # 加载FINER数据作为后备
            train_data = load_finer("data/train.jsonl", max_samples=1)
            if train_data:
                sample = train_data[0]
            else:
                # 如果还是没有数据，则使用默认数据
                sample = {
                    "question": "What is the revenue of company X in Q1 2023?",
                    "context": "Company X reported total revenue of $1.2 billion in Q1 2023, up from $1.0 billion in Q1 2022.",
                    "answer": "1.2 billion"
                }

        # 初始化客户端
        client = GLMClient(model="glm-4.6v")

        # 创建生成器节点
        generator = GeneratorNode(client, task_type="finer")

        # 生成提示词
        prompt = get_generator_prompt(
            question=sample.get("question", sample.get("text", "")),
            playbook={},
            task_type="finer",
            context=sample.get("context", "")
        )

        print(f"✅ FINER提示词生成成功")
        print(f"提示词预览: {prompt[:200]}...")

        return True
    except Exception as e:
        print(f"❌ FINER提示词生成失败: {e}")
        return False

def test_finer_generator(train_data=None):
    """测试FINER生成器"""
    print("\n=== 测试FINER生成器 ===")

    try:
        # 加载环境变量
        load_env()

        # 使用训练数据中的样本，如果未提供则使用默认值
        if train_data and len(train_data) > 0:
            sample = train_data[0]  # 使用第一个训练样本
        else:
            # 加载FINER数据作为后备
            train_data = load_finer("data/train.jsonl", max_samples=1)
            if train_data:
                sample = train_data[0]
            else:
                # 如果还是没有数据，则使用默认数据
                sample = {
                    "question": "What is the revenue of company X in Q1 2023?",
                    "context": "Company X reported total revenue of $1.2 billion in Q1 2023, up from $1.0 billion in Q1 2022.",
                    "answer": "1.2 billion"
                }

        # 初始化客户端
        client = GLMClient(model="glm-4.6v")

        # 创建生成器节点
        generator = GeneratorNode(client, task_type="finer")

        # 测试生成
        result = generator({
            "current_playbook": {},
            "current_sample": sample,
            "ground_truth": sample.get("answer", sample.get("label", ""))
        })

        print(f"✅ FINER生成器测试成功")
        print(f"生成答案: {result.get('generated_answer', 'N/A')}")
        print(f"推理过程: {result.get('generator_trace', 'N/A')}")

        return True
    except Exception as e:
        print(f"❌ FINER生成器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FINER任务测试")
    print("=" * 50)

    # 测试数据加载
    train_data, val_data = test_finer_data_loading()

    # 测试提示词生成，传入训练数据
    test_finer_prompt_generation(train_data)

    # 测试生成器，传入训练数据
    test_finer_generator(train_data)

    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()