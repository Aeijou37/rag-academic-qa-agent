"""
Reranker 模块 — Cross-Encoder 精排

核心设计：
双塔模型（bge embedding）的 query 和 doc 是独立编码的，精度有上限。
Cross-Encoder（bge-reranker）把 query 和 doc 拼在一起送入模型，能捕捉
细粒度的语义交互，精排效果显著优于双塔。

流程：
  query → bge 向量检索 → Top-N 候选 → Reranker 精排 → Top-K 最终结果

技术选型：BAAI/bge-reranker-large（和 bge embedding 同家族，兼容性好）
"""
import torch
import numpy as np
from typing import List, Dict, Optional


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-large"


class Reranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        """加载 Cross-Encoder Reranker 模型"""
        print(f"加载 Reranker 模型: {self.model_name}")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()
            print("Reranker 加载完成")
        except Exception as e:
            print(f"⚠️ Reranker 加载失败: {e}")
            print("  精排将不可用，回退到纯向量检索")
            self.model = None
            self.tokenizer = None

    def is_available(self) -> bool:
        """Reranker 是否可用"""
        return self.model is not None

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 4,
    ) -> List[Dict]:
        """对候选 chunk 做精排

        Args:
            query: 用户查询
            candidates: 向量检索返回的候选 chunk 列表
            top_k: 精排后返回的数量

        Returns:
            精排后的 top_k chunk 列表（带 rerank_score）
        """
        if not self.is_available() or len(candidates) == 0:
            return candidates[:top_k]

        if len(candidates) <= top_k:
            return candidates

        pairs = []
        for c in candidates:
            content = c["content"][:500]
            pairs.append([query, content])

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits.squeeze(-1)
        scores = logits.cpu().numpy()

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        reranked = []
        for idx in ranked_indices:
            item = candidates[idx].copy()
            item["rerank_score"] = float(scores[idx])
            item["vector_score"] = item.get("score", 0)
            reranked.append(item)

        return reranked

    @torch.no_grad()
    def score_pair(self, query: str, document: str) -> float:
        """对单个 query-document 对打分"""
        if not self.is_available():
            return 0.0

        inputs = self.tokenizer(
            [[query, document[:500]]],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits.squeeze(-1)
        return float(logits.cpu().numpy()[0])
