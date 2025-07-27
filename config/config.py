from pymilvus import MilvusClient


# ==== Mlivus 配置区 ====
DB_FILE = "./data/rag_local.db"
COLLECTION_NAME = "rag_documents"
client = MilvusClient(DB_FILE)


# ==== API 配置区 ====
DIMENSION = 1024
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
CHAT_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-nhnklopejonbklumkchlnsjaluxbetocvqdzevgcrjptjlpj"