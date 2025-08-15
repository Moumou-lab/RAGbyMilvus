from typing import Optional

from config.config import *
from loguru import logger
from pymilvus import MilvusClient


_client: Optional[MilvusClient] = None


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        logger.info("初始化 Milvus 客户端 …")
        _client = MilvusClient(DB_FILE)
    return _client


def ensure_collection(dimension: Optional[int] = None) -> MilvusClient:
    """确保集合存在并已加载，必要时创建集合。"""
    client = get_client()
    if not client.has_collection(COLLECTION_NAME):
        dim = int(dimension or DIMENSION)
        logger.info(f"集合 {COLLECTION_NAME} 不存在，正在创建… (dim={dim})")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dim,
            auto_id=True,
        )
    else:
        logger.info(f"集合 {COLLECTION_NAME} 已存在")
    client.load_collection(COLLECTION_NAME)
    return client


def milvus_client_init(dimension: Optional[int] = None) -> MilvusClient:
    return ensure_collection(dimension)
