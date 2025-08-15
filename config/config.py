import os


# ==== Milvus 配置 ====
# 可用环境变量覆盖：RAG_DB_FILE, RAG_COLLECTION_NAME, RAG_EMBEDDING_DIM
DB_FILE = os.getenv("RAG_DB_FILE", "./data/rag_local.db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_documents")
# 注意：请确保该维度与实际嵌入模型一致
DIMENSION = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))


# ==== API 配置 ====
EMBEDDING_API_URL = os.getenv(
    "RAG_EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings"
)
CHAT_API_URL = os.getenv(
    "RAG_CHAT_API_URL", "https://api.siliconflow.cn/v1/chat/completions"
)
# 从环境变量读取敏感信息，避免泄露到代码库
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
