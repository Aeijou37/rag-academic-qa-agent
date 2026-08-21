"""
评估框架 — RAG 系统的量化评估

把技术笔记里的定性消融实验升级为可运行的量化评估。

评估维度：
1. 检索质量
   - Recall@K: 答案所在 chunk 是否被检索到
   - MRR (Mean Reciprocal Rank): 答案所在 chunk 的排名
2. 生成质量
   - Faithfulness: 回答是否忠于检索内容（不编造）
   - Answer Relevance: 回答是否切题
3. 引用准确率
   - Citation Accuracy: 来源标注是否正确

评估数据格式（data/eval_dataset.json）:
[
  {
    "question": "这篇论文的主要贡献是什么？",
    "answer_chunks": ["chunk_id_1", "chunk_id_2"],  // 标注的答案所在 chunk
    "expected_keywords": ["贡献", "创新", "方法"],   // 答案应包含的关键词
    "source": "paper.pdf",                           // 答案来源文档
    "page": 3                                        // 答案来源页码
  }
]
"""
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_4: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0

@dataclass
class GenerationMetrics:
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    citation_accuracy: float = 0.0
    instruction_leakage_rate: float = 0.0
    verbatim_copy_rate: float = 0.0

@dataclass
class EvalReport:
    total_questions: int = 0
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    per_question: List[Dict] = field(default_factory=list)


class RAGEvaluator:
    def __init__(self, eval_dataset_path: str = "data/eval_dataset.json"):
        self.eval_data = self._load_dataset(eval_dataset_path)
        self.report = EvalReport()

    def _load_dataset(self, path: str) -> List[Dict]:
        """加载评估数据集"""
        p = Path(path)
        if not p.exists():
            print(f"⚠️ 评估数据集不存在: {path}")
            print("  请创建评估数据集后运行评估")
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"评估数据集已加载: {len(data)} 个问题")
        return data

    def evaluate_retrieval(
        self,
        rag_chain,
        methods: List[str] = None,
    ) -> Dict[str, RetrievalMetrics]:
        """评估检索质量

        对每种检索方法计算 Recall@K 和 MRR

        Args:
            rag_chain: RAGChain 实例
            methods: 要评估的检索方法列表

        Returns:
            {method_name: RetrievalMetrics} 字典
        """
        if not self.eval_data:
            return {}

        if methods is None:
            methods = ["similarity", "mmr", "rewrite"]

        results = {}

        for method in methods:
            print(f"\n评估检索方法: {method}")
            print("-" * 40)

            recalls_1 = []
            recalls_4 = []
            recalls_10 = []
            reciprocal_ranks = []

            for i, item in enumerate(self.eval_data):
                question = item["question"]
                answer_chunks = set(item.get("answer_chunks", []))
                answer_source = item.get("source", "")
                answer_page = item.get("page", "")

                try:
                    retrieved = rag_chain.retriever.search(
                        question, top_k=10, method=method
                    )
                except Exception as e:
                    print(f"  Q{i+1} 检索失败: {e}")
                    continue

                hit = False
                for rank, chunk in enumerate(retrieved):
                    meta = chunk["metadata"]
                    chunk_source = meta.get("source", "")
                    chunk_page = meta.get("page", meta.get("paragraph", ""))

                    is_hit = (
                        chunk_source == answer_source
                        and str(chunk_page) == str(answer_page)
                    ) or len(answer_chunks) == 0

                    if is_hit and not hit:
                        rank_pos = rank + 1
                        reciprocal_ranks.append(1.0 / rank_pos)
                        if rank_pos <= 1:
                            recalls_1.append(1.0)
                        if rank_pos <= 4:
                            recalls_4.append(1.0)
                        if rank_pos <= 10:
                            recalls_10.append(1.0)
                        hit = True

                if not hit:
                    recalls_1.append(0.0)
                    recalls_4.append(0.0)
                    recalls_10.append(0.0)
                    reciprocal_ranks.append(0.0)

            metrics = RetrievalMetrics(
                recall_at_1=np.mean(recalls_1) if recalls_1 else 0.0,
                recall_at_4=np.mean(recalls_4) if recalls_4 else 0.0,
                recall_at_10=np.mean(recalls_10) if recalls_10 else 0.0,
                mrr=np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            )

            results[method] = metrics
            print(f"  Recall@1:  {metrics.recall_at_1:.2%}")
            print(f"  Recall@4:  {metrics.recall_at_4:.2%}")
            print(f"  Recall@10: {metrics.recall_at_10:.2%}")
            print(f"  MRR:       {metrics.mrr:.4f}")

        return results

    def evaluate_generation(self, rag_chain) -> GenerationMetrics:
        """评估生成质量

        指标：
        - Faithfulness: 回答中的关键词是否出现在检索内容中
        - Answer Relevance: 回答是否包含预期关键词
        - Citation Accuracy: 来源标注是否正确
        - Instruction Leakage: 回答是否泄露 System Prompt
        - Verbatim Copy: 回答是否大段复制原文
        """
        if not self.eval_data:
            return GenerationMetrics()

        print(f"\n评估生成质量")
        print("-" * 40)

        faithfulness_scores = []
        relevance_scores = []
        citation_scores = []
        leakage_count = 0
        copy_count = 0

        leaked_keywords = [
            "你是一个", "请遵守", "System Prompt", "system prompt",
            "你必须", "以下规则", "严格遵守",
        ]

        for i, item in enumerate(self.eval_data):
            question = item["question"]
            expected_keywords = item.get("expected_keywords", [])
            expected_source = item.get("source", "")
            expected_page = item.get("page", "")

            result = rag_chain.query(question)

            answer = result["answer"]
            retrieved = result["retrieved_chunks"]
            context = "\n".join(c["content"] for c in retrieved)

            relevance = self._calc_relevance(answer, expected_keywords)
            faithfulness = self._calc_faithfulness(answer, context)
            citation_correct = self._check_citation(
                answer, expected_source, expected_page
            )
            has_leakage = any(kw in answer for kw in leaked_keywords)
            has_copy = self._check_verbatim_copy(answer, context)

            faithfulness_scores.append(faithfulness)
            relevance_scores.append(relevance)
            citation_scores.append(1.0 if citation_correct else 0.0)

            if has_leakage:
                leakage_count += 1
            if has_copy:
                copy_count += 1

            print(f"  Q{i+1}: relevance={relevance:.2f} faithfulness={faithfulness:.2f} "
                  f"citation={'✅' if citation_correct else '❌'}")

        total = len(self.eval_data)
        metrics = GenerationMetrics(
            faithfulness=np.mean(faithfulness_scores),
            answer_relevance=np.mean(relevance_scores),
            citation_accuracy=np.mean(citation_scores),
            instruction_leakage_rate=leakage_count / total,
            verbatim_copy_rate=copy_count / total,
        )

        print(f"\n生成质量汇总:")
        print(f"  Faithfulness:          {metrics.faithfulness:.2%}")
        print(f"  Answer Relevance:      {metrics.answer_relevance:.2%}")
        print(f"  Citation Accuracy:     {metrics.citation_accuracy:.2%}")
        print(f"  Instruction Leakage:   {metrics.instruction_leakage_rate:.2%}")
        print(f"  Verbatim Copy Rate:    {metrics.verbatim_copy_rate:.2%}")

        return metrics

    def _calc_relevance(self, answer: str, expected_keywords: List[str]) -> float:
        """计算答案相关性：预期关键词命中率"""
        if not expected_keywords:
            return 1.0
        hits = sum(1 for kw in expected_keywords if kw in answer)
        return hits / len(expected_keywords)

    def _calc_faithfulness(self, answer: str, context: str) -> float:
        """计算忠实度：回答中的关键内容是否能在检索内容中找到依据"""
        sentences = re.split(r"[。.！!？?；;]", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return 0.0

        supported = 0
        for sent in sentences:
            words = [w for w in jieba_cut(sent) if len(w) > 1]
            if not words:
                continue

            context_words = set(jieba_cut(context))
            word_hits = sum(1 for w in words if w in context_words)
            coverage = word_hits / len(words)

            if coverage > 0.5:
                supported += 1

        return supported / len(sentences)

    def _check_citation(self, answer: str, expected_source: str, expected_page) -> bool:
        """检查来源标注是否正确"""
        if expected_source and expected_source not in answer:
            return False
        if expected_page and str(expected_page) not in answer:
            return False
        return True

    def _check_verbatim_copy(self, answer: str, context: str) -> bool:
        """检测是否大段复制原文（>50字连续匹配）"""
        for i in range(len(context) - 50):
            snippet = context[i:i+50]
            if snippet in answer:
                return True
        return False

    def run_full_evaluation(self, rag_chain) -> Dict:
        """运行完整评估（检索 + 生成）"""
        print("\n" + "=" * 60)
        print("RAG 系统完整评估")
        print("=" * 60)

        retrieval_results = self.evaluate_retrieval(rag_chain)
        generation_metrics = self.evaluate_generation(rag_chain)

        report = {
            "total_questions": len(self.eval_data),
            "retrieval": {
                method: {
                    "recall_at_1": m.recall_at_1,
                    "recall_at_4": m.recall_at_4,
                    "recall_at_10": m.recall_at_10,
                    "mrr": m.mrr,
                }
                for method, m in retrieval_results.items()
            },
            "generation": {
                "faithfulness": generation_metrics.faithfulness,
                "answer_relevance": generation_metrics.answer_relevance,
                "citation_accuracy": generation_metrics.citation_accuracy,
                "instruction_leakage_rate": generation_metrics.instruction_leakage_rate,
                "verbatim_copy_rate": generation_metrics.verbatim_copy_rate,
            },
        }

        self._print_summary(report)
        return report

    def _print_summary(self, report: Dict):
        """打印评估汇总"""
        print("\n" + "=" * 60)
        print("评估汇总")
        print("=" * 60)
        print(f"问题数: {report['total_questions']}")

        print(f"\n检索质量:")
        for method, m in report["retrieval"].items():
            print(f"  {method:12s} | R@1={m['recall_at_1']:.2%} | "
                  f"R@4={m['recall_at_4']:.2%} | R@10={m['recall_at_10']:.2%} | "
                  f"MRR={m['mrr']:.4f}")

        print(f"\n生成质量:")
        g = report["generation"]
        print(f"  Faithfulness:        {g['faithfulness']:.2%}")
        print(f"  Answer Relevance:    {g['answer_relevance']:.2%}")
        print(f"  Citation Accuracy:   {g['citation_accuracy']:.2%}")
        print(f"  Instruction Leakage: {g['instruction_leakage_rate']:.2%}")
        print(f"  Verbatim Copy:       {g['verbatim_copy_rate']:.2%}")

    def save_report(self, report: Dict, output_path: str = "results/eval_report.json"):
        """保存评估报告"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n评估报告已保存: {output_path}")


def jieba_cut(text: str) -> List[str]:
    """中文分词（优先用 jieba，fallback 到字符切分）"""
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        return list(text)
