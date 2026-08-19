"""
检索模块 — MMR 检索 + 查询改写 + 图文混合检索

多模态 RAG 检索策略：
1. 文本查询 → bge编码 → 在所有chunk中检索（文本+图像描述都能匹配）
2. 图像查询 → CLIP编码 → 用CLIP向量检索图像chunk
3. MMR：相似度 + 多样性约束
4. 查询改写：双路取并集
"""
from typing import List, Dict, Optional
import numpy as np


class Retriever:
    def __init__(self, vector_store, llm=None, image_processor=None, lambda_mmr: float = 0.7):
        self.vs = vector_store
        self.llm = llm
        self.image_processor = image_processor
        self.lambda_mmr = lambda_mmr

    def search(self, query: str, top_k: int = 4, method: str = "mmr", image_path: str = None) -> List[Dict]:
        """检索接口（支持文本或图像查询）"""
        if image_path and self.image_processor and self.image_processor.is_available():
            return self.image_search(image_path, top_k)

        if method == "similarity":
            return self.vs.search(query, top_k)
        elif method == "mmr":
            return self.mmr_search(query, top_k)
        elif method == "rewrite":
            return self.rewrite_search(query, top_k)
        else:
            return self.mmr_search(query, top_k)

    def image_search(self, image_path: str, top_k: int = 4) -> List[Dict]:
        """图像查询：用CLIP编码图像，检索相关chunk"""
        if not self.image_processor or not self.image_processor.is_available():
            print("⚠️ CLIP不可用，无法执行图像检索")
            return []

        print(f"  图像检索: {image_path}")
        clip_emb = self.image_processor.encode_image(image_path)
        if clip_emb is None:
            return []

        results = self.vs.collection.query(
            query_embeddings=[clip_emb],
            n_results=top_k,
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
            })
        print(f"  图像检索到 {len(docs)} 个结果")
        return docs

    def mmr_search(self, query: str, top_k: int = 4, fetch_k: int = 20) -> List[Dict]:
        """MMR 检索：相似度 + 多样性"""
        candidates = self.vs.search(query, top_k=fetch_k)
        if len(candidates) <= top_k:
            return candidates

        query_emb = self.vs._embed_query(query)
        doc_embs = [self.vs._embed_query(c["content"]) for c in candidates]

        selected = []
        selected_indices = []
        remaining = list(range(len(candidates)))

        for _ in range(top_k):
            best_idx = None
            best_score = -float("inf")

            for idx in remaining:
                sim_query = self._cosine_sim(query_emb, doc_embs[idx])

                if selected_indices:
                    sim_selected = max(self._cosine_sim(doc_embs[idx], doc_embs[s]) for s in selected_indices)
                else:
                    sim_selected = 0

                mmr_score = self.lambda_mmr * sim_query - (1 - self.lambda_mmr) * sim_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(candidates[best_idx])
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def rewrite_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """查询改写 + 双路取并集"""
        if self.llm is None:
            return self.mmr_search(query, top_k)

        rewritten = self._rewrite_query(query)
        print(f"  原始查询: {query}")
        print(f"  改写查询: {rewritten}")

        results_original = self.mmr_search(query, top_k=top_k)
        results_rewritten = self.mmr_search(rewritten, top_k=top_k)

        merged = self._merge_dedupe(results_original, results_rewritten, top_k)
        return merged

    def _rewrite_query(self, query: str) -> str:
        """用 LLM 改写查询"""
        prompt = (
            f"请将以下学术问题改写为更适合文档检索的关键词形式，"
            f"提取核心概念和术语，输出一段简洁的检索词（不超过50字）：\n\n{query}"
        )
        try:
            rewritten = self.llm.generate(prompt, max_new_tokens=50)
            return rewritten.strip().strip('"').strip("'")
        except Exception as e:
            print(f"  查询改写失败，使用原始查询: {e}")
            return query

    def _merge_dedupe(self, list1: List[Dict], list2: List[Dict], top_k: int) -> List[Dict]:
        """合并两路检索结果并去重"""
        seen = set()
        merged = []
        for item in list1 + list2:
            key = item["content"][:100]
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged[:top_k]

    @staticmethod
    def _cosine_sim(a, b) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
