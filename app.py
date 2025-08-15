import os
import re
import sys
from typing import List, Optional
from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.config import *
from client_init import milvus_client_init
from my_models.chat_model import chat_generate
from my_models.embed_model import get_embedding

# ==== 初始化 Milvus Lite ====
# logger.add(sink=sys.stderr, level="INFO", format="{time} | {level} | {message}")
logger.info("初始化 Milvus 客户端...")
client = milvus_client_init()

def _unique_preserve(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ==== 文本数据上传至向量数据库中（按大段落切分） ====
def document_to_milvus(file_path: str, overlap_ratio: float = 0.0):
    logger.info(f"[LoadFile-Para] 加载文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 正则匹配, 两个段落以上则划分
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
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

    # 去重与清洗（同一文本不重复插入；过滤空白）
    chunks = [c.strip() for c in chunks if c and c.strip()]
    chunks = _unique_preserve(chunks)
    logger.info(f"[LoadFile-Para] 去重后 chunk 数: {len(chunks)}")

    embeddings = get_embedding(chunks)
    to_insert = [{"vector": emb, "text": txt} for emb, txt in zip(embeddings, chunks)]
    client.insert(collection_name=COLLECTION_NAME, data=to_insert)
    client.flush(COLLECTION_NAME)
    logger.success(f"[LoadFile-Para] 成功插入 {len(to_insert)} 条记录")


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
    # 结果去重，避免返回重复内容影响回答多样性
    texts_unique = _unique_preserve([t for t in texts if t])[:top_k]
    logger.info(f"[Search] 命中文档数: {len(texts_unique)}（去重前 {len(texts)}）")
    return texts_unique


# ==== FastAPI 接口 ====
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class DocInput(BaseModel):
    text: str


class IngestFileSpec(BaseModel):
    file_path: str
    overlap_ratio: Optional[float] = 0.0

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
    # 单条插入（不推荐）：也可以 reuse document_to_milvus
    chunks = [doc.text.strip()]
    embeddings = get_embedding(chunks)
    to_insert = [{"vector": embeddings[0], "text": chunks[0]}]
    client.insert(collection_name=COLLECTION_NAME, data=to_insert)
    client.flush(COLLECTION_NAME)
    return {"status": "ok"}

@app.post("/ingest_file")
def ingest_file(spec: IngestFileSpec):
    """接收参数 {file_path: str, overlap_ratio: float} 并进行入库。"""
    fp = spec.file_path
    ov = spec.overlap_ratio or 0.0
    try:
        document_to_milvus(fp, overlap_ratio=ov)
        return {"status": "ok", "chunks": "done"}
    except Exception as e:
        logger.error(f"[API] /ingest_file 错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ==== 本地测试函数 ====
def test_local_rag(data_path: str):
    logger.info("==== 本地测试开始 ====")
    try:
        document_to_milvus(data_path, overlap_ratio=0.0)
        while True:
            query = input("请输入查询内容 (/bye 退出): ").strip()
            if query.lower() == "/bye":
                logger.info("==== 测试结束 ====")
                break
            context = search(query=query, top_k=3)
            logger.info("=== 检索片段 ===")
            for i, doc in enumerate(context, 1):
                logger.info(f"[{i}] {doc}")
            answer = chat_generate(query, context)
            logger.success("=== 最终回答 ===\n" + answer)
    except Exception as e:
        logger.error(f"[Test] 本地测试失败: {e}")
        logger.info("==== 本地测试结束 ====")


if __name__ == "__main__":
    data_path = "data/masters_ecnu.md"
    test_local_rag(data_path)
