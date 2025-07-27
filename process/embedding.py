from loguru import logger
from typing import List

from config.config import *
from my_models.embed_model import get_embedding

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

    logger.info(f"[LoadFile-Para] 总 chunk 数（含重叠）: {len(chunks)}")

    embeddings = get_embedding(chunks)
    to_insert = [{"vector": emb, "text": txt} for emb, txt in zip(embeddings, chunks)]
    client.insert(collection_name=COLLECTION_NAME, data=to_insert)
    client.flush(COLLECTION_NAME)
    logger.success(f"[LoadFile-Para] 成功插入 {len(to_insert)} 条记录")



if __name__ == "__main__":
    test_path = "RAGbyMilvus/data/test_masters.md"
    document_to_milvus(test_path, 0)