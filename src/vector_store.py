"""
向量存储模块 — 图文混合 Chroma 持久化

多模态 RAG 的核心设计：
1. 文本 chunk → bge-large-zh Embedding → 存入 Chroma
2. 图像 chunk → CLIP Embedding（图像描述用bge编码用于检索，CLIP向量存metadata用于跨模态）
3. 统一在一个 Chroma 集合中，用 metadata.modality 区分文本/图像

检索策略：
- 文本查询 → bge 编码 → 在所有chunk中检索（文本+图像描述都能匹配）
- 图像查询 → CLIP 编码 → 用CLIP向量检索图像chunk
"""
from typing import List, Dict, Optional
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"


class VectorStore:
    def __init__(
        self,
        persist_dir: str = "./data/vector_db",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.collection = None
        self._client = None
        self._init_embeddings()

    def _init_embeddings(self):
        """初始化文本 Embedding 模型（bge）"""
        print(f"加载文本 Embedding 模型: {self.embedding_model_name}")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer(self.embedding_model_name)
        print("Embedding 模型加载完成")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """文本转向量（bge）"""
        if hasattr(self.embeddings, "embed_documents"):
            return self.embeddings.embed_documents(texts)
        else:
            return self.embeddings.encode(texts, normalize_embeddings=True).tolist()

    def _embed_query(self, query: str) -> List[float]:
        """查询文本转向量（bge）"""
        if hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(query)
        else:
            return self.embeddings.encode([query], normalize_embeddings=True).tolist()[0]

    def create(self, chunks: List[Dict], image_processor=None):
        """从文档块创建向量数据库（支持图文混合）"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._client.get_or_create_collection(
            name="academic_docs",
            metadata={"hnsw:space": "cosine"},
        )

        text_chunks = [c for c in chunks if c["metadata"].get("modality", "text") == "text"]
        image_chunks = [c for c in chunks if c["metadata"].get("modality") == "image"]

        print(f"入库: {len(text_chunks)} 文本 + {len(image_chunks)} 图像")

        if text_chunks:
            texts = [c["content"] for c in text_chunks]
            metadatas = [c["metadata"] for c in text_chunks]
            ids = [f"txt_{m.get('doc_id', 'x')}_{m.get('chunk_index', i)}" for i, m in enumerate(metadatas)]
            embeddings = self._embed_texts(texts)
            self._batch_upsert(embeddings, texts, metadatas, ids)

        if image_chunks and image_processor and image_processor.is_available():
            for i, chunk in enumerate(image_chunks):
                desc = chunk.get("image_description", chunk["content"])
                clip_emb = chunk.get("clip_embedding")

                text_embedding = self._embed_texts([desc])[0]

                meta = chunk["metadata"].copy()
                if clip_emb:
                    meta["clip_embedding_dim"] = len(clip_emb)

                img_id = f"img_{meta.get('doc_id', 'x')}_{meta.get('page', i)}_{i}"
                self.collection.upsert(
                    embeddings=[text_embedding],
                    documents=[f"[图像描述] {desc}"],
                    metadatas=[meta],
                    ids=[img_id],
                )

        print(f"向量数据库创建完成: {self.collection.count()} 个chunk → {self.persist_dir}")

    def _batch_upsert(self, embeddings, texts, metadatas, ids, batch_size=100):
        """批量upsert"""
        for i in range(0, len(texts), batch_size):
            self.collection.upsert(
                embeddings=embeddings[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

    def load(self):
        """加载已有的向量数据库"""
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._client.get_or_create_collection(
            name="academic_docs",
            metadata={"hnsw:space": "cosine"},
        )
        count = self.collection.count()
        print(f"已加载向量数据库: {count} 个chunk")
        return count

    def search(self, query: str, top_k: int = 4, modality: str = None) -> List[Dict]:
        """检索（支持按模态过滤）"""
        query_embedding = self._embed_query(query)

        where_filter = None
        if modality:
            where_filter = {"modality": modality}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
            })
        return docs

    def search_with_filter(self, query: str, top_k: int = 4, doc_source: str = None) -> List[Dict]:
        """带文档过滤的检索"""
        query_embedding = self._embed_query(query)
        where_filter = {"source": doc_source} if doc_source else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
            })
        return docs

    def list_documents(self) -> List[str]:
        """列出数据库中所有文档名"""
        all_meta = self.collection.get()
        sources = set()
        for m in all_meta["metadatas"]:
            if "source" in m:
                sources.add(m["source"])
        return sorted(sources)

    def delete_document(self, doc_source: str):
        """删除某篇文档的所有chunk"""
        self.collection.delete(where={"source": doc_source})
        print(f"已删除: {doc_source}")

    def count(self) -> int:
        return self.collection.count() if self.collection else 0

    def get_stats(self) -> Dict:
        """获取数据库统计（按模态分类）"""
        if not self.collection:
            return {"total": 0, "text": 0, "image": 0}
        all_meta = self.collection.get()
        text_count = sum(1 for m in all_meta["metadatas"] if m.get("modality", "text") == "text")
        image_count = sum(1 for m in all_meta["metadatas"] if m.get("modality") == "image")
        return {"total": self.collection.count(), "text": text_count, "image": image_count}
