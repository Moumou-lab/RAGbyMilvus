from config.config import *
from loguru import logger
from pymilvus import MilvusClient



def milvus_client_init():

    client = MilvusClient(DB_FILE)
    if not client.has_collection(COLLECTION_NAME):
        logger.info(f"集合 {COLLECTION_NAME} 不存在，正在创建...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=DIMENSION,
            auto_id=True
        )
    else:
        logger.info(f"集合 {COLLECTION_NAME} 已存在")

    client.load_collection(COLLECTION_NAME)