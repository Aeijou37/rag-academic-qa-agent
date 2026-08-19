"""
RAG 学术问答 Agent — 主入口

用法:
  python main.py --model_path ./models/qwen2.5-7b-chat --mode cli
  python main.py --model_path ./models/qwen2.5-7b-chat --mode web
  python main.py --mode cli --no_model  (只检索不生成)

完整流程:
  1. 上传文档 → 解析 → 切分 → 向量化 → Chroma存储
  2. 提问 → 查询改写 → MMR检索 → chat_template构建 → LLM生成 → 后处理 → 回答+来源
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.vector_store import VectorStore
from src.image_processor import ImageProcessor
from src.retriever import Retriever
from src.rag_chain import RAGChain
from src.utils import print_separator, format_sources


def build_knowledge_base(data_dir: str, persist_dir: str, vlm_path: str = None, load_4bit: bool = False):
    """构建多模态知识库"""
    print_separator("构建多模态知识库")

    loader = DocumentLoader()
    loader.load_directory(data_dir)
    print(f"文档统计: {loader.get_stats()}")

    splitter = TextSplitter()
    chunks = splitter.split(loader.documents)

    vlm_for_images = None
    if vlm_path:
        from src.vlm import VLMWrapper
        vlm_for_images = VLMWrapper(vlm_path, load_in_4bit=load_4bit)

    image_processor = ImageProcessor(vlm_wrapper=vlm_for_images)
    if image_processor.is_available():
        chunks = image_processor.process_chunks(chunks)

    vs = VectorStore(persist_dir=persist_dir)
    vs.create(chunks, image_processor=image_processor)

    print_separator("知识库构建完成")
    return vs


def run_cli(model_path: str, load_4bit: bool, persist_dir: str, data_dir: str, vlm_path: str = None):
    """命令行交互模式"""
    print_separator("多模态 RAG 学术问答 Agent (CLI)")

    vs = VectorStore(persist_dir=persist_dir)
    count = vs.load()

    if count == 0:
        if Path(data_dir).exists():
            vs = build_knowledge_base(data_dir, persist_dir, vlm_path=vlm_path, load_4bit=load_4bit)
        else:
            print(f"向量库为空且数据目录不存在: {data_dir}")
            print("请先上传文档构建知识库，或指定 --data_dir")
            return

    llm = None
    if model_path:
        from src.llm import LLMWrapper
        llm = LLMWrapper(model_path, load_in_4bit=load_4bit)
    else:
        print("⚠️ 未指定生成模型，只做检索不生成")

    vlm_for_images = None
    if vlm_path:
        from src.vlm import VLMWrapper
        vlm_for_images = VLMWrapper(vlm_path, load_in_4bit=load_4bit)

    image_processor = ImageProcessor(vlm_wrapper=vlm_for_images)
    retriever = Retriever(vs, llm=llm, image_processor=image_processor)
    rag = RAGChain(retriever, llm=llm)

    print_separator("开始问答")
    print("输入问题开始问答，输入 'quit' 退出\n")

    history = []
    while True:
        try:
            question = input("📝 问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue

        result = rag.query(question, history)
        history.append({"question": question, "answer": result["answer"]})

        print(f"\n💬 回答:\n{result['answer']}")
        print(f"\n📚 来源:\n{format_sources(result['retrieved_chunks'])}")
        print(f"\n{'─' * 60}\n")

    print("\n再见！")


def run_web(model_path: str, load_4bit: bool, persist_dir: str, vlm_path: str = None):
    """Web 界面模式"""
    from src.app import RAGApp
    app = RAGApp(model_path=model_path, vlm_path=vlm_path, load_4bit=load_4bit, persist_dir=persist_dir)
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态 RAG 学术问答 Agent")
    parser.add_argument("--model_path", type=str, default=None, help="生成模型路径")
    parser.add_argument("--vlm_path", type=str, default=None, help="VLM路径（图像描述）")
    parser.add_argument("--load_4bit", action="store_true", help="4bit 量化加载")
    parser.add_argument("--mode", type=str, default="cli", choices=["cli", "web"], help="运行模式")
    parser.add_argument("--data_dir", type=str, default="./docs", help="文档目录")
    parser.add_argument("--persist_dir", type=str, default="./data/vector_db", help="向量库持久化目录")
    parser.add_argument("--no_model", action="store_true", help="不加载生成模型（只检索）")
    args = parser.parse_args()

    if args.no_model:
        model_path = None
    else:
        model_path = args.model_path

    if args.mode == "web":
        run_web(model_path, args.load_4bit, args.persist_dir, args.vlm_path)
    else:
        run_cli(model_path, args.load_4bit, args.persist_dir, args.data_dir, args.vlm_path)
