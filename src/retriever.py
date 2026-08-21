"""
检索模块 — 完整检索策略

支持的检索方法：
1. similarity    — 纯向量相似度检索
2. mmr           — MMR（相似度 + 多样性）
3. rewrite       — 查询改写 + 双路取并集
4. rerank        — 向量初筛 + Reranker 精排
5. hybrid        — 混合检索（BM25 + 向量 + RRF 融合）+ Reranker 精排
6. hyde          — HyDE（假设性文档嵌入）+ MMR + Reranker 精排
7. full          — 查询改写 + 混合检索 + HyDE + Reranker（全部叠加）

检索流程（以 full 为例）：
  query
    ↓ 查询改写（LLM 改写为关键词）
    ↓ HyDE（LLM 生成假设回答）
    ↓ BM25 检索 + 向量检索（原始query + 改写query + 假设回答 三路）
    ↓ RRF 融合
    ↓ MMR 去冗余
    ↓ Reranker 精排
    ↓ Top-K
"""
from typing import List, Dict, Optional
import numpy as np


class Retriever:
    def __init__(
        self,
        vector_store,
        llm=None,
        image_processor=None,
        reranker=None,
        bm25_retriever=None,
        lambda_mmr: float = 0.7,
        use_reranker: bool = True,
    ):
        self.vs = vector_store
        self.llm = llm
        self.image_processor = image_processor
        self.reranker = reranker
        self.bm25 = bm25_retriever
        self.lambda_mmr = lambda_mmr
        self.use_reranker = use_reranker and (reranker is not None) and reranker.is_available()

    def search(self, query: str, top_k: int = 4, method: str = "mmr", image_path: str = None) -> List[Dict]:
        """检索接口（支持多种检索策略）"""
        if image_path and self.image_processor and self.image_processor.is_available():
            return self.image_search(image_path, top_k)

        if method == "similarity":
            results = self.vs.search(query, top_k)
        elif method == "mmr":
            results = self.mmr_search(query, top_k)
        elif method == "rewrite":
            results = self.rewrite_search(query, top_k)
        elif method == "rerank":
            results = self.rerank_search(query, top_k)
        elif method == "hybrid":
            results = self.hybrid_search(query, top_k)
        elif method == "hyde":
            results = self.hyde_search(query, top_k)
        elif method == "full":
            results = self.full_search(query, top_k)
        else:
            results = self.mmr_search(query, top_k)

        return results

    # ==================== 基础检索方法 ====================

    def image_search(self, image_path: str, top_k: int = 4) -> List[Dict]:
        """图像查询：用CLIP编码图像，检索相关chunk"""
        if not self.image_processor or not self.image_processor.is_available():
            print("  CLIP 不可用，无法执行图像检索")
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

    def rerank_search(self, query: str, top_k: int = 4, fetch_k: int = 20) -> List[Dict]:
        """Reranker 精排检索：向量检索 Top-N → Cross-Encoder 精排 Top-K"""
        candidates = self.vs.search(query, top_k=fetch_k)
        print(f"  向量检索候选: {len(candidates)} 个")

        if not self.use_reranker:
            print("  Reranker 不可用，回退到 MMR")
            return self.mmr_search(query, top_k)

        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        print(f"  Reranker 精排完成: Top-{len(reranked)}")
        return reranked

    # ==================== 高级检索方法 ====================

    def hybrid_search(self, query: str, top_k: int = 4, fetch_k: int = 20) -> List[Dict]:
        """混合检索：BM25 关键词检索 + 向量检索 → RRF 融合 → Reranker 精排

        向量检索擅长语义匹配，BM25 擅长字面匹配（专有名词/缩写/公式编号）。
        RRF 融合两路结果，取长补短。
        """
        print("  混合检索: BM25 + 向量")

        vector_results = self.vs.search(query, top_k=fetch_k)
        print(f"    向量检索: {len(vector_results)} 个候选")

        bm25_results = []
        if self.bm25 is not None:
            bm25_results = self.bm25.search(query, top_k=fetch_k)
            print(f"    BM25 检索: {len(bm25_results)} 个候选")

        if not bm25_results:
            if self.use_reranker:
                return self.reranker.rerank(query, vector_results, top_k=top_k)
            return vector_results[:top_k]

        from src.bm25_retriever import rrf_fusion
        fused = rrf_fusion(vector_results, bm25_results, top_k=fetch_k)
        print(f"    RRF 融合: {len(fused)} 个候选")

        if self.use_reranker:
            fused = self.reranker.rerank(query, fused, top_k=top_k)
            print(f"    Reranker 精排: Top-{len(fused)}")

        return fused[:top_k]

    def hyde_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """HyDE：假设性文档嵌入

        让 LLM 先生成一个"假设性回答"，用这个回答去检索。
        假设回答和文档的语义空间更接近（都是"回答"而非"问题"），
        检索命中率更高。

        流程: query → LLM生成假设回答 → 用假设回答做MMR检索 → Reranker精排
        """
        if self.llm is None:
            print("  HyDE 需要 LLM，回退到 MMR")
            return self.mmr_search(query, top_k)

        hypothetical_doc = self._generate_hypothetical_answer(query)
        print(f"  HyDE 假设回答: {hypothetical_doc[:100]}...")

        results = self.mmr_search(hypothetical_doc, top_k=top_k * 2)

        if self.use_reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)
            print(f"  Reranker 精排: Top-{len(results)}")

        return results[:top_k]

    def full_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """全部优化叠加：查询改写 + HyDE + 混合检索 + RRF 融合 + Reranker 精排

        最完整的检索策略，把所有优化都叠加：
        1. LLM 改写 query
        2. LLM 生成假设回答（HyDE）
        3. 对 原始query + 改写query + 假设回答 分别做向量检索 + BM25检索
        4. RRF 融合所有结果
        5. Reranker 精排
        """
        print("  Full Search: 改写 + HyDE + 混合检索 + RRF + Reranker")

        all_vector_results = []
        all_bm25_results = []

        all_vector_results.extend(self.vs.search(query, top_k=20))

        if self.llm is not None:
            rewritten = self._rewrite_query(query)
            print(f"    改写: {rewritten}")
            all_vector_results.extend(self.vs.search(rewritten, top_k=20))

            hypothetical = self._generate_hypothetical_answer(query)
            print(f"    HyDE: {hypothetical[:80]}...")
            all_vector_results.extend(self.vs.search(hypothetical, top_k=20))

        if self.bm25 is not None:
            all_bm25_results.extend(self.bm25.search(query, top_k=20))
            if self.llm is not None:
                all_bm25_results.extend(self.bm25.search(rewritten, top_k=20))

        from src.bm25_retriever import rrf_fusion

        if all_bm25_results:
            fused = rrf_fusion(all_vector_results, all_bm25_results, top_k=20)
        else:
            fused = self._merge_dedupe(all_vector_results, [], top_k=20)

        print(f"    融合候选: {len(fused)} 个")

        if self.use_reranker:
            fused = self.reranker.rerank(query, fused, top_k=top_k)
            print(f"    Reranker 精排: Top-{len(fused)}")

        return fused[:top_k]

    # ==================== 辅助方法 ====================

    def _rewrite_query(self, query: str) -> str:
        """用 LLM 改写查询为检索关键词"""
        prompt = (
            f"请将以下学术问题改写为更适合文档检索的关键词形式，"
            f"提取核心概念和术语，输出一段简洁的检索词（不超过50字）：\n\n{query}"
        )
        try:
            rewritten = self.llm.generate(prompt, max_new_tokens=50)
            return rewritten.strip().strip('"').strip("'")
        except Exception as e:
            print(f"    查询改写失败: {e}")
            return query

    def _generate_hypothetical_answer(self, query: str) -> str:
        """HyDE：让 LLM 生成假设性回答用于检索"""
        prompt = (
            f"请针对以下学术问题，写一段100字左右的假设性回答。"
            f"不需要完全准确，但要包含可能相关的术语和概念，用于文档检索：\n\n{query}"
        )
        try:
            answer = self.llm.generate(prompt, max_new_tokens=150)
            return answer.strip()
        except Exception as e:
            print(f"    HyDE 生成失败: {e}")
            return query

    def _apply_reranker(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        """对检索结果应用 Reranker 精排"""
        if not self.use_reranker or len(results) <= 1:
            return results

        print(f"  Reranker 精排: {len(results)} 候选 → Top-{top_k}")
        return self.reranker.rerank(query, results, top_k=top_k)

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
