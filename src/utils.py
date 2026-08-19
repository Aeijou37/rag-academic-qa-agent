"""
工具函数模块
"""
from pathlib import Path
from typing import Dict, List
import hashlib


def ensure_dir(dir_path: str) -> Path:
    """确保目录存在"""
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_hash(file_path: str) -> str:
    """计算文件MD5哈希（用于判断文件是否已处理）"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def format_sources(retrieved: List[Dict]) -> str:
    """格式化来源溯源信息"""
    sources = []
    seen = set()
    for chunk in retrieved:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "未知")
        page = meta.get("page", meta.get("paragraph", meta.get("chunk_index", "")))
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"📄 {source} (位置: {page})")
    return "\n".join(sources) if sources else "未找到来源"


def truncate_text(text: str, max_len: int = 200) -> str:
    """截断文本"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def print_separator(title: str = ""):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)
