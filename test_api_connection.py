"""
API连接测试脚本 - 用于定位main.py中的API调用问题
"""

import os
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm.glm_client import GLMClient
from src.utils.env import load_env

def test_client_initialization():
    """测试API客户端初始化"""
    print("=== 测试API客户端初始化 ===")

    try:
        # 加载环境变量
        load_env()

        # 初始化客户端
        client = GLMClient(
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            model="glm-4.6v"
        )
        print("✅ 客户端初始化成功")
        return client
    except Exception as e:
        print(f"❌ 客户端初始化失败: {e}")
        return None

def test_connection_check(client):
    """测试API连接检查"""
    print("\n=== 测试API连接检查 ===")

    try:
        result = client.check_api_connection()
        if result:
            print("✅ API连接检查通过")
        else:
            print("❌ API连接检查失败")
        return result
    except Exception as e:
        print(f"❌ 连接检查异常: {e}")
        return False

def test_basic_call(client):
    """测试基本API调用"""
    print("\n=== 测试基本API调用 ===")

    try:
        response = client.call(
            prompt="你好，请回复'OK'",
            temperature=0.0,
            max_tokens=10
        )
        print(f"✅ 基本调用成功，响应: {response}")
        return True
    except Exception as e:
        print(f"❌ 基本调用失败: {e}")
        return False

def test_call_with_messages(client):
    """测试带消息的API调用"""
    print("\n=== 测试带消息的API调用 ===")

    try:
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手"},
            {"role": "user", "content": "你好，请回复'OK'"}
        ]

        response = client.call_with_messages(
            messages=messages,
            temperature=0.0,
            max_tokens=10
        )
        print(f"✅ 带消息调用成功，响应: {response}")
        return True
    except Exception as e:
        print(f"❌ 带消息调用失败: {e}")
        return False

def test_json_call(client):
    """测试JSON响应调用"""
    print("\n=== 测试JSON响应调用 ===")

    try:
        result = client.call_json(
            prompt="返回一个包含status字段的JSON对象",
            temperature=0.0
        )
        print(f"✅ JSON调用成功，结果: {result}")
        return True
    except Exception as e:
        print(f"❌ JSON调用失败: {e}")
        return False

def main():
    """主测试函数"""
    print("API连接问题诊断测试")
    print("=" * 50)

    # 测试客户端初始化
    client = test_client_initialization()
    if not client:
        return

    # 测试连接检查
    test_connection_check(client)

    # 测试各种API调用
    test_basic_call(client)
    test_call_with_messages(client)
    test_json_call(client)

    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == "__main__":
    main()