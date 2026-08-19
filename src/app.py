"""
Gradio 前端 — 多模态 RAG 学术问答 Agent

功能：
1. 上传文档（PDF/DOCX/TXT/MD）→ 自动解析+切分+图片提取+向量化
2. 文本提问 或 图像提问 → 检索+生成回答+来源溯源
3. 显示检索到的原文片段 + 相关图像
4. 多轮对话

运行:
  python src/app.py --model_path ./models/qwen2.5-7b-chat
  python src/app.py --model_path ./models/qwen2.5-7b-chat --load_4bit
  python src/app.py --model_path ./models/qwen2.5-7b-chat --vlm_path ./models/qwen-vl-chat
"""
import sys
import gradio as gr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter
from src.vector_store import VectorStore
from src.image_processor import ImageProcessor
from src.retriever import Retriever
from src.rag_chain import RAGChain
from src.llm import LLMWrapper


class RAGApp:
    def __init__(
        self,
        model_path: str = None,
        vlm_path: str = None,
        load_4bit: bool = False,
        persist_dir: str = "./data/vector_db",
    ):
        self.model_path = model_path
        self.vlm_path = vlm_path
        self.load_4bit = load_4bit
        self.persist_dir = persist_dir
        self.loader = DocumentLoader()
        self.splitter = TextSplitter()
        self.vs = None
        self.image_processor = None
        self.retriever = None
        self.rag = None
        self.llm = None
        self.vlm = None
        self.chat_history = []
        self._init_system()

    def _init_system(self):
        """初始化多模态 RAG 系统"""
        print("初始化多模态 RAG 系统...")

        self.vs = VectorStore(persist_dir=self.persist_dir)
        count = self.vs.load()

        if self.model_path:
            print(f"加载生成模型: {self.model_path}")
            self.llm = LLMWrapper(self.model_path, load_in_4bit=self.load_4bit)
        else:
            print("⚠️ 未指定生成模型，将只做检索不生成")
            self.llm = None

        vlm_for_images = None
        if self.vlm_path:
            print(f"加载 VLM（图像描述）: {self.vlm_path}")
            from src.vlm import VLMWrapper
            vlm_for_images = VLMWrapper(self.vlm_path, load_in_4bit=self.load_4bit)

        self.image_processor = ImageProcessor(vlm_wrapper=vlm_for_images)
        if not self.image_processor.is_available():
            print("⚠️ CLIP不可用，图像检索将禁用（文本检索不受影响）")

        self.retriever = Retriever(self.vs, llm=self.llm, image_processor=self.image_processor)
        self.rag = RAGChain(self.retriever, llm=self.llm)
        print("系统初始化完成\n")

    def upload_docs(self, files):
        """上传文档并构建多模态向量库"""
        if not files:
            return "请上传文档"

        results = []
        for file in files:
            if isinstance(file, str):
                path = file
            else:
                path = file.name

            try:
                docs = self.loader.load(path)
                results.append(f"✅ {Path(path).name}: {len(docs)} 块")
            except Exception as e:
                results.append(f"❌ {Path(path).name}: {e}")

        chunks = self.splitter.split(self.loader.documents)

        if self.image_processor:
            chunks = self.image_processor.process_chunks(chunks)

        self.vs.create(chunks, image_processor=self.image_processor)

        stats = self.loader.get_stats()
        summary = "\n".join(results)
        summary += f"\n\n总计: {stats['total_chunks']} 块"
        summary += f"\n  文本: {stats['modalities'].get('text', 0)}"
        summary += f"\n  图像: {stats['modalities'].get('image', 0)}"
        summary += f"\n  文档数: {stats['total_documents']}"
        summary += f"\n向量库: {self.vs.count()} 个chunk"
        return summary

    def ask(self, question: str, query_image=None):
        """问答（支持文本+图像查询）"""
        if not question and not query_image:
            return "请输入问题或上传查询图像", "", "", None

        if self.vs.count() == 0:
            return "请先上传文档", "", "", None

        image_path = query_image if isinstance(query_image, str) else None
        if hasattr(query_image, 'name'):
            image_path = query_image.name

        result = self.rag.query(question or "请描述与这张图相关的内容", image_path=image_path)

        self.chat_history.append({
            "question": question or "[图像查询]",
            "answer": result["answer"],
        })

        sources = result["sources"]

        retrieved_text = ""
        for i, chunk in enumerate(result["retrieved_chunks"], 1):
            meta = chunk["metadata"]
            source = meta.get("source", "未知")
            page = meta.get("page", meta.get("paragraph", ""))
            modality = meta.get("modality", "text")
            content = chunk["content"][:150]
            tag = "📷图像" if modality == "image" else "📝文本"
            retrieved_text += f"[{tag}{i}] {source} ({page})\n{content}...\n\n"

        retrieved_images = [c for c in result["retrieved_chunks"] if c["metadata"].get("modality") == "image"]
        first_image = None
        if retrieved_images:
            img_path = retrieved_images[0]["metadata"].get("image_name")
            if img_path:
                full_path = Path("./data/extracted_images") / img_path
                if full_path.exists():
                    first_image = str(full_path)

        return result["answer"], sources, retrieved_text, first_image

    def build(self):
        with gr.Blocks(title="多模态 RAG 学术问答 Agent") as demo:
            gr.Markdown("# 多模态 RAG 学术问答 Agent")
            gr.Markdown("上传学术文档，用**文本或图像**提问，系统检索相关内容（含图表）并生成带来源溯源的回答。")

            with gr.Tab("文档管理"):
                gr.Markdown("### 上传文档")
                gr.Markdown("支持 PDF / DOCX / TXT / MD。PDF 中的图表会被自动提取并生成描述。")
                file_input = gr.Files(label="上传文档", file_types=[".pdf", ".docx", ".txt", ".md"])
                upload_btn = gr.Button("处理文档", variant="primary")
                upload_output = gr.Textbox(label="处理结果", lines=10, interactive=False)

                upload_btn.click(
                    fn=self.upload_docs,
                    inputs=[file_input],
                    outputs=[upload_output],
                )

            with gr.Tab("问答"):
                with gr.Row():
                    question_input = gr.Textbox(
                        label="文本提问",
                        placeholder="例如：论文里的实验结果图说明了什么？",
                        lines=2,
                    )
                    image_input = gr.Image(label="图像提问（可选）", type="filepath")

                ask_btn = gr.Button("提问", variant="primary")

                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Textbox(label="回答", lines=10, interactive=False)
                    with gr.Column(scale=1):
                        sources_output = gr.Textbox(label="来源溯源", lines=5, interactive=False)
                        retrieved_image_output = gr.Image(label="检索到的相关图像", interactive=False)

                retrieved_output = gr.Textbox(label="检索到的原文片段", lines=12, interactive=False)

                ask_btn.click(
                    fn=self.ask,
                    inputs=[question_input, image_input],
                    outputs=[answer_output, sources_output, retrieved_output, retrieved_image_output],
                )

            with gr.Tab("帮助"):
                gr.Markdown("""
                ### 使用说明

                **文档管理**：
                1. 上传 PDF/DOCX/TXT/MD 文件
                2. 系统自动提取文本 + 图片
                3. 图片用 VLM 生成描述，用 CLIP 编码入库

                **问答**：
                - **文本提问**：输入问题，系统检索相关文本和图像描述
                - **图像提问**：上传图片，系统用 CLIP 检索相关内容
                - 回答末尾标注信息来源

                ### 多模态技术

                | 技术 | 作用 |
                |---|---|
                | PDF图片提取 | 从论文中提取图表/架构图 |
                | VLM图像描述 | 把图像转为文本描述，让LLM理解 |
                | CLIP图文编码 | 跨模态检索（文本↔图像） |
                | MMR检索 | 相似度+多样性，避免重复 |
                | 查询改写 | 双路取并集，提升命中率 |
                | chat_template | 标准化格式，防止指令泄露 |
                | 强约束Prompt+后处理 | 控制原文复述和幻觉 |
                """)

        return demo


if __name__ == "__main__":
    args = sys.argv[1:]
    model_path = None
    vlm_path = None
    load_4bit = False
    persist_dir = "./data/vector_db"

    for i, arg in enumerate(args):
        if arg == "--model_path" and i + 1 < len(args):
            model_path = args[i + 1]
        elif arg == "--vlm_path" and i + 1 < len(args):
            vlm_path = args[i + 1]
        elif arg == "--load_4bit":
            load_4bit = True
        elif arg == "--persist_dir" and i + 1 < len(args):
            persist_dir = args[i + 1]

    app = RAGApp(model_path=model_path, vlm_path=vlm_path, load_4bit=load_4bit, persist_dir=persist_dir)
    demo = app.build()
    demo.launch(server_name="0.0.0.0", server_port=7860)
