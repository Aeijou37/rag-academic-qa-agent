# RAG Academic QA Agent

A local academic document question-answering agent based on **Retrieval-Augmented Generation (RAG)**, **LangChain**, **Chroma**, and local LLM deployment.

Upload academic documents → ask questions in natural language → get answers with source tracing. Fully local, no external API needed.

---

## 📌 Project Overview

This project builds a local academic knowledge base QA system using RAG technology. The system supports document parsing, semantic text chunking, vector embedding, persistent vector storage, MMR retrieval, query rewriting, and local question answering with source tracing.

**Key insight**: In RAG systems, **retrieval quality > generation quality > prompt engineering** — the upstream bottleneck (retrieval) determines the system's ceiling. This is structurally identical to my industrial defect classification work (data strategy > model enhancement > loss function).

---

## ✨ Key Features

- **Multi-format support**: PDF / DOCX / TXT / MD
- **Semantic chunking**: RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
- **MMR retrieval**: Similarity + diversity constraint, avoids redundant chunks
- **Query rewriting**: LLM rewrites query for better retrieval (dual-path merge with original)
- **chat_template**: Standardized chat format to prevent instruction leakage in small models
- **Constrained System Prompt + post-processing**: Controls verbatim copying and hallucination
- **Source tracing**: Every answer includes source document and page/paragraph reference
- **Multi-document management**: Cross-document retrieval with metadata filtering
- **Local deployment**: No external API required, privacy-friendly
- **Gradio web interface**: Upload documents, ask questions, view retrieved passages

---

## 🧠 Technical Stack

| Component | Selection | Purpose |
|---|---|---|
| Document parsing | PyPDF2 / python-docx / markdown | PDF/DOCX/TXT/MD |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Semantic-first chunking |
| Embedding | BAAI/bge-large-zh-v1.5 | Chinese semantic retrieval |
| Vector database | Chroma | Persistent storage with metadata |
| Retrieval | MMR + Query Rewriting | Similarity + diversity + dual-path |
| Generation | Qwen2.5-7B-Chat (FP16/4bit) | Local LLM |
| Chat format | tokenizer.apply_chat_template | Prevent instruction leakage |
| Frontend | Gradio | Web interface |

---

## 🏗️ System Pipeline

```text
Documents (PDF/DOCX/TXT/MD)
   ↓
Document Parsing (extract text + metadata)
   ↓
Semantic Chunking (RecursiveCharacterTextSplitter, 800/150)
   ↓
Embedding (bge-large-zh-v1.5)
   ↓
Chroma Vector Database (persistent, with metadata)
   ↓
Query Rewriting (LLM rewrites query, dual-path merge)
   ↓
MMR Retrieval (similarity + diversity, Top-K)
   ↓
chat_template Construction (standardized format)
   ↓
Constrained System Prompt + Context + Question
   ↓
Local LLM Generation (Qwen2.5-7B-Chat)
   ↓
Post-processing (remove leakage, detect copying, add sources)
   ↓
Answer with Source References
```

---

## 📁 Project Structure

```text
rag-academic-qa-agent/
├── README.md
├── requirements.txt
├── main.py                    # 主入口 (CLI / Web 模式)
├── configs/                   # 配置文件
├── docs/                      # 示例文档
├── examples/                  # 示例
└── src/
    ├── document_loader.py     # 文档加载 (PDF/DOCX/TXT/MD)
    ├── text_splitter.py       # 语义切分
    ├── vector_store.py        # Chroma 向量存储
    ├── retriever.py           # MMR检索 + 查询改写
    ├── rag_chain.py           # RAG链路 (Prompt + 后处理)
    ├── llm.py                 # 本地LLM加载
    ├── app.py                 # Gradio前端
    └── utils.py               # 工具函数
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download embedding model

```bash
# bge-large-zh-v1.5 will auto-download on first run, or pre-download:
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('BAAI/bge-large-zh-v1.5')
"
```

### 3. Download generation model (optional)

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-7B-Chat', local_dir='./models/qwen2.5-7b-chat')
"
```

### 4. Put documents in ./docs/

Place PDF/DOCX/TXT/MD files in the `docs/` directory.

### 5. Run

```bash
# CLI mode (with generation model)
python main.py --model_path ./models/qwen2.5-7b-chat --mode cli

# CLI mode (retrieval only, no generation model)
python main.py --mode cli --no_model

# Web mode (Gradio interface)
python main.py --model_path ./models/qwen2.5-7b-chat --mode web

# Web mode (4bit quantization for 16GB GPU)
python main.py --model_path ./models/qwen2.5-7b-chat --mode web --load_4bit
```

### 6. Use the web interface

Open `http://localhost:7860` in your browser:
1. Upload documents in "文档管理" tab
2. Ask questions in "问答" tab
3. View retrieved passages and source tracing

---

## 📊 Key Design Decisions

### 1. chat_template for instruction leakage prevention

Small models (<7B) leak system prompt content when given plain text. Using `tokenizer.apply_chat_template` with proper role tokens (`<|im_start|>system/user/assistant`) eliminates this issue completely.

### 2. Constrained System Prompt + post-processing for verbatim copying

Small models tend to copy retrieved text verbatim instead of summarizing. Dual protection:
- **Prompt layer**: "用自己的语言概括，不要大段复制原文"
- **Post-processing**: Detect >50 char verbatim copies and replace with `[原文概括：...]`

### 3. Query rewriting with dual-path merge

User asks "这个方法怎么解决长尾问题" but the paper says "LDAM margin". Query rewriting:
- Rewritten path: "LDAM margin" → precise match
- Original path: original query → fallback recall
- Merge + deduplicate → best of both

### 4. MMR for retrieval diversity

Pure similarity search returns redundant chunks (e.g., 3 copies of the same abstract). MMR selects chunks that are both relevant to the query AND different from already-selected chunks: `λ * Sim(query, doc) - (1-λ) * max(Sim(selected, doc))`.

---

## 👤 Author

**Guojie Li**

Master's Student in Information and Communication Engineering
Research interests: Computer Vision, Industrial AI, 3D Reconstruction, RAG Agent, Local LLM Deployment

GitHub: [Aeijou37](https://github.com/Aeijou37)
