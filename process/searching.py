
from typing import List
from loguru import logger

from config.config import *
from my_models.embed_model import get_embedding

# ==== 检索文档 ====
def search(query: str, top_k: int = 3) -> List[str]:
    logger.info(f"[Search] 查询: {query}")
    embedding = get_embedding([query])[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding],
        output_fields=["text"],
        limit=top_k,
        params={"metric_type": "COSINE", "params": {"nprobe": 10}}
    )
    hits = results[0]
    texts = [hit.entity.get("text") for hit in hits]
    logger.info(f"[Search] 命中文档数: {len(texts)}")
    return texts