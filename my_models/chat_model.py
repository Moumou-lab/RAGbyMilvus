import json
import requests
from typing import List, Optional
from loguru import logger

from config.config import *




# ==== RAG 模型回答 ====
def chat_generate(query: str, context: List[str]) -> str:
    logger.info(f"[RAG] 发送提问: {query}")
    if not API_KEY:
        raise RuntimeError("SILICONFLOW_API_KEY 未配置，请设置环境变量后重试")
    doc_text = "\n".join(context)
    system_prompt = """你是 “师大先生”，一个检索增强生成（RAG）助手。
严格基于用户提供文档回答。
禁止引入外部信息。
回答格式简洁明了。
"""

    user_prompt = f"""
文档内容：
{doc_text}
用户问题：
{query}
请基于以上文档内容简洁回答：
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen3-8B",
        "max_tokens": 512,
        "enable_thinking": False,
        "thinking_budget": 4096,
        "min_p": 0,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "frequency_penalty": 0.5,
        "n": 1, # 生成数量
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        resp = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        logger.success("[RAG] 获取回答成功")
        return result
    except Exception as e:
        logger.error(f"[RAG] 大模型调用失败: {e}")
        raise RuntimeError("大模型调用失败")

