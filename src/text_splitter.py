"""
文本切分模块 — 语义优先切分

策略：RecursiveCharacterTextSplitter，按段落/标题优先切分
参数：chunk_size=800, overlap=150（中文按字符计）

关键设计：学术文档有天然的语义边界（章节/段落/标题），按语义切分而非固定字符数。
"""
from typing import List, Dict


DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

CHUNK_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",
    "\n\n\n", "\n\n", "\n",
    "。", "！", "？", "；",
    ".", "!", "?", ";",
    " ", "",
]


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[Dict]) -> List[Dict]:
        """切分文档块列表"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=CHUNK_SEPARATORS,
                length_function=len,
            )
        except ImportError:
            return self._fallback_split(documents)

        chunks = []
        for doc in documents:
            texts = splitter.split_text(doc["content"])
            for i, text in enumerate(texts):
                chunk = {
                    "content": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunk_total": len(texts),
                    },
                }
                chunks.append(chunk)

        print(f"切分完成: {len(documents)} 个文档块 → {len(chunks)} 个chunk (size={self.chunk_size}, overlap={self.overlap})")
        return chunks

    def _fallback_split(self, documents: List[Dict]) -> List[Dict]:
        """无 LangChain 时的简易切分"""
        chunks = []
        for doc in documents:
            text = doc["content"]
            if len(text) <= self.chunk_size:
                chunks.append({**doc, "metadata": {**doc["metadata"], "chunk_index": 0}})
                continue
            start = 0
            idx = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": idx},
                })
                start = end - self.overlap
                idx += 1
        print(f"切分完成(fallback): {len(documents)} → {len(chunks)} chunks")
        return chunks
