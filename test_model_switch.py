"""
模型切换测试脚本
用于验证不同GLM模型的切换功能
"""

import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm.glm_client import GLMClient
from src.utils.env import load_env

def test_model_initialization(model_name: str):
    """测试不同模型的初始化"""
    print(f"=== 测试 {model_name} 初始化 ===")

    try:
        # 加载环境变量
        load_env()

        # 初始化客户端
        client = GLMClient(model=model_name)

        print(f"✅ {model_name} 客户端初始化成功")

        # 测试连接
        if client.check_api_connection():
            print(f"✅ {model_name} API连接成功")
        else:
            print(f"❌ {model_name} API连接失败")

        return True
    except Exception as e:
        print(f"❌ {model_name} 初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("GLM模型切换测试")
    print("=" * 50)

    # 测试不同的模型
    models_to_test = ["glm-4.6v", "glm-4.5-air", "glm-4"]

    for model in models_to_test:
        test_model_initialization(model)
        print()

    print("=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()