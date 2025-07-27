import os
import sys
import json
from typing import List, Optional
from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from pymilvus import MilvusClient

from config.config import *
from my_models.chat_model import chat_generate
from my_models.embed_model import get_embedding

# ==== 初始化 Milvus Lite ====
logger.add(sink=sys.stderr, level="INFO", format="{time} | {level} | {message}")
logger.info("初始化 Milvus 客户端...")

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

# ==== 文本入库（按大段落切分） ====
def load_and_ingest_by_paragraph(file_path: str, overlap_ratio: float = 0.0):
    logger.info(f"[LoadFile-Para] 加载文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 默认按两个换行分段
    raw_paras = [p.strip() for p in content.split("\n\n") if p.strip()]
    logger.info(f"[LoadFile-Para] 原始段落数: {len(raw_paras)}")

    chunks: List[str] = []
    for i, para in enumerate(raw_paras):
        chunks.append(para)
        # 可选 overlap：后一段前部 + 前一段尾部
        if overlap_ratio and i + 1 < len(raw_paras):
            next_para = raw_paras[i + 1]
            tail = para.split()[-int(len(para.split()) * overlap_ratio):]
            head = next_para.split()[:int(len(next_para.split()) * overlap_ratio)]
            overlap_text = " ".join(tail + head)
            chunks.append(overlap_text)

    logger.info(f"[LoadFile-Para] 总 chunk 数（含重叠）: {len(chunks)}")

    embeddings = get_embedding(chunks)
    to_insert = [{"vector": emb, "text": txt} for emb, txt in zip(embeddings, chunks)]
    client.insert(collection_name=COLLECTION_NAME, data=to_insert)
    client.flush(COLLECTION_NAME)
    logger.success(f"[LoadFile-Para] 成功插入 {len(to_insert)} 条记录")


# ==== 检索文档 ====
def search(query: str, top_k: int = 5) -> List[str]:
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


# ==== FastAPI 接口 ====
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class DocInput(BaseModel):
    text: str

@app.post("/rag")
def rag_answer(req: QueryRequest):
    logger.info(f"[API] /rag 请求，query: {req.query!r}")
    context = search(req.query, top_k=req.top_k)
    if not context:
        raise HTTPException(status_code=404, detail="未找到相关文档")
    answer = chat_generate(req.query, context)
    return {"answer": answer, "docs": context}

@app.post("/add_doc")
def add_doc(doc: DocInput):
    logger.info(f"[API] /add_doc 请求文本长度: {len(doc.text)}")
    # 单条插入（不推荐）：也可以 reuse load_and_ingest_by_paragraph
    chunks = [doc.text.strip()]
    embeddings = get_embedding(chunks)
    to_insert = [{"vector": embeddings[0], "text": chunks[0]}]
    client.insert(collection_name=COLLECTION_NAME, data=to_insert)
    client.flush(COLLECTION_NAME)
    return {"status": "ok"}

@app.post("/ingest_file")
def ingest_file(spec: BaseModel):
    """
    接收参数 { “file_path”: str, “overlap_ratio”: float }
    """
    d = spec.dict()
    fp = d.get("file_path")
    ov = d.get("overlap_ratio", 0.0)
    try:
        load_and_ingest_by_paragraph(fp, overlap_ratio=ov)
        return {"status": "ok", "chunks": "done"}
    except Exception as e:
        logger.error(f"[API] /ingest_file 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==== 本地测试函数（可选） ====
def test_local_rag():
    logger.info("==== 本地测试开始 ====")
    try:
        load_and_ingest_by_paragraph("data/masters.md", overlap_ratio=0.0)
        query = "查询曹孚老师"
        context = search(query)
        logger.info("=== 检索片段 ===")
        for i, doc in enumerate(context, 1):
            logger.info(f"[{i}] {doc}")
        answer = chat_generate(query, context)
        logger.success("=== 最终回答 ===\n" + answer)
    except Exception as e:
        logger.error(f"[Test] 本地测试失败: {e}")
    logger.info("==== 本地测试结束 ====")


if __name__ == "__main__":
    test_local_rag()
