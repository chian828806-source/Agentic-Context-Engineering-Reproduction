import os
import time
import json
from typing import Optional, Dict, Any, List
from zai import ZhipuAiClient
from src.utils.env import load_env

class GLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        model: str = "glm-4.6",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ):
        load_env()
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided.")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_calls = 0

        self.client = ZhipuAiClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

    def call(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.call_with_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def call_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if response_format:
                    params["response_format"] = response_format

                response = self.client.chat.completions.create(**params)
                
                # --- 核心修复逻辑 ---
                message_obj = response.choices[0].message
                content = getattr(message_obj, 'content', '') or ''
                reasoning = getattr(message_obj, 'reasoning_content', '') or ''
                
                # 如果标准内容为空但有推理内容，则使用推理内容
                final_res = content if content.strip() else reasoning
                # ------------------

                if hasattr(response, "usage") and response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                    self.total_tokens += response.usage.total_tokens
                self.total_calls += 1

                return final_res

            except Exception as e:
                last_error = e
                if any(x in str(e).lower() for x in ["authentication", "api key"]):
                    raise RuntimeError(f"Auth error: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue

        raise RuntimeError(f"API call failed after retries: {last_error}")

    def call_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = self.call(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
        )
        try:
            # 清理 Markdown 代码块包裹
            clean_json = response.strip()
            if clean_json.startswith("```"):
                clean_json = clean_json.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if clean_json.startswith("json"):
                clean_json = clean_json[4:].strip()
                
            return json.loads(clean_json)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e}, Content: {response[:200]}")

    def get_token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
        }

    def check_api_connection(self) -> bool:
        try:
            # 对于推理模型，要求它给出简短回答
            response = self.call(prompt="Respond with only 'OK'", temperature=0.0, max_tokens=10)
            return "OK" in response.upper()
        except:
            return False

if __name__ == "__main__":
    # 快速测试
    client = GLMClient(model="glm-4.5-air")
    print(f"Connection: {client.check_api_connection()}")
    print(f"Usage: {client.get_token_usage()}")