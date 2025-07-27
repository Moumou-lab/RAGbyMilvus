import json
import requests
from typing import List, Optional
from loguru import logger

from config.config import *




# ==== RAG 模型回答 ====
def chat_generate(query: str, context: List[str]) -> str:
    logger.info(f"[RAG] 发送提问: {query}")
    doc_text = "\n".join(context)
    user_prompt = f"""你是一个智能助手，请基于以下文档内容回答问题：
文档内容：
{doc_text}
用户问题：{query}
请结合文档内容简洁回答：
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
            {"role": "system", "content": "你的名字叫‘师大先生’"},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        resp = requests.post(CHAT_API_URL, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("[RAG] 获取回答成功")
        return result
    except Exception as e:
        logger.error(f"[RAG] 大模型调用失败: {e}")
        raise RuntimeError("大模型调用失败")


