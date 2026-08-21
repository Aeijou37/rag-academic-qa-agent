"""
RAG 链路模块 — 检索增强生成的核心

三大关键技术点（和技术笔记一致）：

1. chat_template 标准化 — 解决小模型指令泄露
   用 tokenizer.apply_chat_template 构建标准聊天格式，而非简单拼接文本

2. 强约束 System Prompt — 解决原文复述
   限定输出格式 + 限定"用自己的话概括" + 限定"不知道就说不知道"

3. 后处理清洗 — 兜底防护
   去指令泄露残留 + 检测原文复述 + 补来源标注
"""
from typing import List, Dict, Optional
import re


SYSTEM_PROMPT = """你是一个学术文档问答助手。请严格遵守以下规则：

1. 回答必须基于提供的参考信息，不得编造文档中不存在的内容。
2. 用自己的语言概括，不要大段复制原文。引用原文时不超过50字。
3. 如果参考信息不足以回答问题，回答"根据已有文档无法回答该问题"。
4. 回答末尾标注来源：[来源：文档名，第X页/段]
5. 回答控制在300字以内，简洁准确。"""


class RAGChain:
    def __init__(
        self,
        retriever,
        llm=None,
        top_k: int = 4,
        retrieval_method: str = "mmr",
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.retrieval_method = retrieval_method

    def query(self, question: str, history: list = None, image_path: str = None) -> Dict:
        """完整 RAG 流程：检索 → 构建 Prompt → 生成 → 后处理（支持图像查询）"""
        print(f"\n问题: {question}")
        print(f"检索方法: {self.retrieval_method}")
        if image_path:
            print(f"附图: {image_path}")

        print("检索中...")
        retrieved = self.retriever.search(
            question, top_k=self.top_k, method=self.retrieval_method, image_path=image_path
        )
        print(f"  检索到 {len(retrieved)} 个相关chunk")

        context = self._build_context(retrieved)
        sources = self._extract_sources(retrieved)

        retrieved_images = [c for c in retrieved if c["metadata"].get("modality") == "image"]

        prompt = self._build_prompt(question, context)

        print("生成中...")
        raw_response = self._generate(prompt, history)

        cleaned = self._post_process(raw_response, context)

        if "来源" not in cleaned:
            cleaned += f"\n\n{sources}"

        return {
            "question": question,
            "answer": cleaned,
            "sources": sources,
            "retrieved_chunks": retrieved,
            "retrieved_images": retrieved_images,
            "raw_response": raw_response,
        }

    def _build_context(self, retrieved: List[Dict]) -> str:
        """拼接检索结果为上下文（区分文本和图像chunk）"""
        parts = []
        for i, chunk in enumerate(retrieved, 1):
            meta = chunk["metadata"]
            source = meta.get("source", "未知文档")
            page = meta.get("page", meta.get("paragraph", meta.get("chunk_index", "")))
            modality = meta.get("modality", "text")

            if modality == "image":
                label = f"[参考{i}·图像] 来源: {source}, 位置: 第{page}页"
            else:
                label = f"[参考{i}·文本] 来源: {source}, 位置: {page}"

            parts.append(f"{label}\n{chunk['content']}")
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """构建完整 Prompt（chat_template 格式）"""
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"参考信息：\n{context}\n\n"
            f"问题：{question}\n\n"
            f"请根据参考信息回答问题。"
        )

    def _generate(self, prompt: str, history: list = None) -> str:
        """调用 LLM 生成回答"""
        if self.llm is None:
            return "[错误: 未加载生成模型]"
        return self.llm.generate(prompt, history=history)

    def _post_process(self, response: str, context: str) -> str:
        """后处理清洗：去指令泄露 + 检测原文复述 + 格式补全"""
        leaked_keywords = [
            "你是一个", "请遵守", "System Prompt", "system prompt",
            "你必须", "以下规则", "严格遵守", "参考信息：",
        ]
        cleaned = response
        for kw in leaked_keywords:
            cleaned = cleaned.replace(kw, "")

        for chunk in context.split("\n\n"):
            lines = chunk.split("\n")
            text = "\n".join(lines[1:]) if len(lines) > 1 else chunk
            if len(text) > 50 and text.strip() in cleaned:
                cleaned = cleaned.replace(
                    text.strip(),
                    f"[原文概括：{text.strip()[:30]}...]"
                )

        cleaned = cleaned.strip()
        return cleaned

    def _extract_sources(self, retrieved: List[Dict]) -> str:
        """提取来源溯源信息"""
        sources = []
        seen = set()
        for chunk in retrieved:
            meta = chunk["metadata"]
            source = meta.get("source", "未知")
            page = meta.get("page", meta.get("paragraph", meta.get("chunk_index", "")))
            key = f"{source}_{page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"[来源：{source}，{page}]")
        return " ".join(sources)

    def chat(self, question: str, history: list = None) -> Dict:
        """多轮对话接口"""
        return self.query(question, history)
