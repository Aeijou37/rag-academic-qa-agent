"""
RAG 学术问答 Agent — 主入口

用法:
  python main.py --model_path ./models/qwen2.5-7b-chat --mode cli
  python main.py --model_path ./models/qwen2.5-7b-chat --mode web
  python main.py --mode cli --no_model  (只检索不生成)
  python main.py --mode cli --retrieval_method hybrid  (混合检索)
  python main.py --mode cli --retrieval_method hyde    (HyDE检索)
  python main.py --mode cli --retrieval_method full     (全部优化叠加)
  python main.py --mode eval --model_path ./models/qwen2.5-7b-chat  (运行评估)

完整流程:
  1. 上传文档 → 解析 → 切分 → 向量化 → Chroma存储
  2. 提问 → [查询改写 → HyDE] → [BM25 + 向量混合检索 → RRF融合] → MMR → Reranker精排 → LLM生成 → 后处理
  3. 评估 → 检索质量(Recall@K, MRR) + 生成质量(Faithfulness, Relevance, Citation)
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
from src.reranker import Reranker
from src.bm25_retriever import BM25Retriever
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
    return vs, chunks


def init_system(model_path, load_4bit, persist_dir, data_dir, vlm_path,
                use_reranker, use_bm25, retrieval_method):
    """初始化 RAG 系统（CLI/Web/Eval 共用）"""
    vs = VectorStore(persist_dir=persist_dir)
    count = vs.load()

    chunks = None
    if count == 0:
        if Path(data_dir).exists():
            vs, chunks = build_knowledge_base(data_dir, persist_dir, vlm_path=vlm_path, load_4bit=load_4bit)
        else:
            print(f"向量库为空且数据目录不存在: {data_dir}")
            return None

    llm = None
    if model_path:
        from src.llm import LLMWrapper
        llm = LLMWrapper(model_path, load_in_4bit=load_4bit)

    vlm_for_images = None
    if vlm_path:
        from src.vlm import VLMWrapper
        vlm_for_images = VLMWrapper(vlm_path, load_in_4bit=load_4bit)

    image_processor = ImageProcessor(vlm_wrapper=vlm_for_images)

    reranker = None
    if use_reranker:
        reranker = Reranker()
        if not reranker.is_available():
            print("  Reranker 不可用")

    bm25 = None
    if use_bm25:
        if chunks is None:
            all_meta = vs.collection.get()
            chunks = []
            for i, doc in enumerate(all_meta["documents"]):
                chunks.append({
                    "content": doc,
                    "metadata": all_meta["metadatas"][i],
                })
        bm25 = BM25Retriever(chunks)

    retriever = Retriever(
        vs, llm=llm, image_processor=image_processor,
        reranker=reranker, bm25_retriever=bm25,
        use_reranker=use_reranker,
    )
    rag = RAGChain(retriever, llm=llm, retrieval_method=retrieval_method)

    return rag


def run_cli(model_path, load_4bit, persist_dir, data_dir, vlm_path,
            use_reranker, use_bm25, retrieval_method):
    """命令行交互模式"""
    print_separator("多模态 RAG 学术问答 Agent (CLI)")

    rag = init_system(model_path, load_4bit, persist_dir, data_dir, vlm_path,
                      use_reranker, use_bm25, retrieval_method)
    if rag is None:
        return

    print_separator("开始问答")
    print(f"检索方法: {retrieval_method}")
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


def run_web(model_path, load_4bit, persist_dir, vlm_path,
            use_reranker, use_bm25, retrieval_method):
    """Web 界面模式"""
    from src.app import RAGApp
    app = RAGApp(
        model_path=model_path, vlm_path=vlm_path,
        load_4bit=load_4bit, persist_dir=persist_dir,
        use_reranker=use_reranker, use_bm25=use_bm25,
        retrieval_method=retrieval_method,
    )
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)


def run_eval(model_path, load_4bit, persist_dir, data_dir, vlm_path,
             use_reranker, use_bm25, retrieval_only, eval_methods):
    """评估模式"""
    print_separator("RAG 系统评估")

    rag = init_system(model_path, load_4bit, persist_dir, data_dir, vlm_path,
                      use_reranker, use_bm25, retrieval_method="mmr")
    if rag is None:
        return

    from src.evaluation import RAGEvaluator
    evaluator = RAGEvaluator(eval_dataset_path="data/eval_dataset.json")

    if len(evaluator.eval_data) == 0:
        print("请先创建评估数据集: data/eval_dataset.json")
        return

    if eval_methods is None:
        eval_methods = ["similarity", "mmr", "rewrite", "hybrid", "hyde", "full"]

    if retrieval_only:
        print(f"\n评估检索方法: {eval_methods}")
        retrieval_results = evaluator.evaluate_retrieval(rag, methods=eval_methods)
    else:
        report = evaluator.run_full_evaluation(rag)
        evaluator.save_report(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态 RAG 学术问答 Agent")
    parser.add_argument("--model_path", type=str, default=None, help="生成模型路径")
    parser.add_argument("--vlm_path", type=str, default=None, help="VLM路径（图像描述）")
    parser.add_argument("--load_4bit", action="store_true", help="4bit 量化加载")
    parser.add_argument("--mode", type=str, default="cli", choices=["cli", "web", "eval"], help="运行模式")
    parser.add_argument("--data_dir", type=str, default="./docs", help="文档目录")
    parser.add_argument("--persist_dir", type=str, default="./data/vector_db", help="向量库持久化目录")
    parser.add_argument("--no_model", action="store_true", help="不加载生成模型（只检索）")
    parser.add_argument("--retrieval_method", type=str, default="mmr",
                        choices=["similarity", "mmr", "rewrite", "rerank", "hybrid", "hyde", "full"],
                        help="检索方法")
    parser.add_argument("--use_reranker", action="store_true", default=True, help="启用 Reranker 精排")
    parser.add_argument("--no_reranker", action="store_true", help="禁用 Reranker 精排")
    parser.add_argument("--no_bm25", action="store_true", help="禁用 BM25 混合检索")
    parser.add_argument("--eval_retrieval_only", action="store_true", help="只评估检索质量")
    parser.add_argument("--eval_methods", type=str, nargs="*", default=None, help="评估的检索方法列表")
    args = parser.parse_args()

    if args.no_model:
        model_path = None
    else:
        model_path = args.model_path

    use_reranker = not args.no_reranker
    use_bm25 = not args.no_bm25

    if args.mode == "web":
        run_web(model_path, args.load_4bit, args.persist_dir, args.vlm_path,
                use_reranker, use_bm25, args.retrieval_method)
    elif args.mode == "eval":
        run_eval(model_path, args.load_4bit, args.persist_dir, args.data_dir, args.vlm_path,
                 use_reranker, use_bm25, args.eval_retrieval_only, args.eval_methods)
    else:
        run_cli(model_path, args.load_4bit, args.persist_dir, args.data_dir, args.vlm_path,
                use_reranker, use_bm25, args.retrieval_method)
