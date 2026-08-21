"""
BM25 检索模块 — 关键词检索 + RRF 融合

为什么需要 BM25：
向量检索擅长语义匹配（"长尾问题" → "LDAM"），
但对专有名词、缩写、公式编号等"字面匹配"场景不友好。
BM25 擅长字面匹配，两者互补。

融合策略：RRF（Reciprocal Rank Fusion）
  对每个 chunk，在两路检索结果中的排名取倒数后求和：
  RRF_score = 1/(k + rank_bm25) + 1/(k + rank_vector)
  k 通常取 60

流程：
  query → BM25 检索 Top-N + 向量检索 Top-N → RRF 融合 → Top-K
"""
import math
import numpy as np
from typing import List, Dict, Optional


class BM25Retriever:
    def __init__(self, chunks: List[Dict], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            chunks: 文档块列表（和向量库中的 chunk 一致）
            k1: 词频饱和参数（控制词频的影响上限）
            b: 文档长度归一化参数（0=不归一化，1=完全归一化）
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_ids = [c["content"][:100] for c in chunks]
        self.doc_tokens = [self._tokenize(c["content"]) for c in chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = np.mean(self.doc_len) if self.doc_len else 1.0
        self.N = len(chunks)

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（优先 jieba，fallback 到字符切分）"""
        try:
            import jieba
            return [w for w in jieba.cut(text) if len(w.strip()) > 0]
        except ImportError:
            return list(text)

    def _build_index(self):
        """构建 BM25 索引：词频表 + 逆文档频率"""
        print(f"构建 BM25 索引: {self.N} 个文档")

        self.tf = []
        self.df = {}
        self.idf = {}

        for tokens in self.doc_tokens:
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.tf.append(tf)

            for token in tf:
                self.df[token] = self.df.get(token, 0) + 1

        for token, df in self.df.items():
            self.idf[token] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

        print(f"  词汇表大小: {len(self.df)}")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词检索"""
        query_tokens = self._tokenize(query)

        scores = []
        for i in range(self.N):
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue

                tf = self.tf[i].get(token, 0)
                if tf == 0:
                    continue

                idf = self.idf[token]
                norm = 1 - self.b + self.b * (self.doc_len[i] / self.avg_doc_len)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

            scores.append(score)

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            if scores[idx] > 0:
                item = self.chunks[idx].copy()
                item["bm25_score"] = float(scores[idx])
                item["bm25_rank"] = rank + 1
                results.append(item)

        return results


def rrf_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 4,
    k: int = 60,
) -> List[Dict]:
    """RRF 融合：向量检索 + BM25 检索结果融合

    Args:
        vector_results: 向量检索结果列表
        bm25_results: BM25 检索结果列表
        top_k: 融合后返回数量
        k: RRF 平滑参数（通常取 60）

    Returns:
        融合后的 top_k chunk 列表
    """
    rrf_scores = {}

    for rank, item in enumerate(vector_results):
        key = item["content"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_results):
        key = item["content"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    seen = {}
    for item in vector_results + bm25_results:
        key = item["content"][:100]
        if key not in seen:
            seen[key] = item

    ranked_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in ranked_keys[:top_k]:
        item = seen[key].copy()
        item["rrf_score"] = rrf_scores[key]
        results.append(item)

    return results
