import json
import requests
from typing import List, Optional
from loguru import logger

from config.config import *




# ==== 嵌入生成 ====
def get_embedding(texts: List[str]) -> List[List[float]]:
    """
    批量生成嵌入向量。texts 是字符串列表。
    """
    logger.info(f"[Embedding] 批量生成嵌入，数量: {len(texts)}")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "input": texts,
        "encoding_format": "float"
    }
    resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    items = resp.json().get("data", [])
    embeddings = [item["embedding"] for item in items]
    logger.info(f"[Embedding] 返回嵌入向量数量: {len(embeddings)}")
    return embeddings
