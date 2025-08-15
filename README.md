# RAGbyMilvus
使用 Milvus Lite 作为向量数据库，实现 RAG。

## 准备
- 创建 `data/` 目录用于存放本地 Milvus Lite 数据库：
  - `mkdir -p data`
- 安装依赖：
  - `pip install -r requirements.txt`
- 配置环境变量（建议放入 `.env` 或启动前在 shell 中导出）：
  - `export SILICONFLOW_API_KEY=your_api_key`
  - 可选：
    - `export RAG_DB_FILE=./data/rag_local.db`
    - `export RAG_COLLECTION_NAME=rag_documents`
    - `export RAG_EMBEDDING_DIM=1024`  # 请与实际嵌入模型维度保持一致

## 运行 API 服务
- 开发运行：
  - `uvicorn app:app --reload --port 8000`

## 使用示例
- 导入文件：
  - `POST /ingest_file`，Body: `{ "file_path": "data/xxx.md", "overlap_ratio": 0.0 }`
- RAG 问答：
  - `POST /rag`，Body: `{ "query": "你的问题", "top_k": 3 }`

## 说明
- 配置项现从环境变量读取，避免在代码中硬编码敏感信息。
- Milvus 客户端在首次启动时自动初始化并确保集合存在。
