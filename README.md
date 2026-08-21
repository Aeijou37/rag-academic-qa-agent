# 多模态 RAG Academic QA Agent

A **multimodal** academic document question-answering agent with **hybrid retrieval**, **Reranker**, **HyDE**, and **quantitative evaluation**. Based on **RAG**, **LangChain**, **Chroma**, **CLIP**, **BM25**, and local LLM/VLM deployment.

Upload academic documents → ask questions with **text or image** → get answers with source tracing. PDF charts/figures are automatically extracted and indexed. Fully local, no external API needed.

---

## 📌 Project Overview

This project builds a local academic knowledge base QA system using RAG technology. The system supports multimodal document parsing, semantic text chunking, hybrid retrieval (BM25 + vector), MMR, query rewriting, HyDE, Cross-Encoder Reranker, and local question answering with source tracing.

**Key insight**: In RAG systems, **retrieval quality > generation quality > prompt engineering** — the upstream bottleneck (retrieval) determines the system's ceiling. This is structurally identical to my industrial defect classification work (data strategy > model enhancement > loss function).

---

## ✨ Key Features

- **Multimodal**: PDF image extraction + VLM description + CLIP cross-modal encoding
- **Hybrid retrieval**: BM25 keyword + bge vector → RRF fusion (covers both literal and semantic matching)
- **MMR**: Similarity + diversity constraint, avoids redundant chunks
- **Query rewriting**: LLM rewrites query for better retrieval (dual-path merge)
- **HyDE**: LLM generates hypothetical answer, uses it for retrieval (closer to document semantic space)
- **Reranker**: Cross-Encoder (bge-reranker-large) for precision re-ranking
- **chat_template**: Standardized chat format to prevent instruction leakage
- **Post-processing**: Detect verbatim copying + remove instruction leakage + add source references
- **Evaluation framework**: Recall@K, MRR, Faithfulness, Answer Relevance, Citation Accuracy
- **7 retrieval methods**: similarity / mmr / rewrite / rerank / hybrid / hyde / full
- **Source tracing**: Every answer includes document name + page/paragraph reference
- **Local deployment**: No external API required, privacy-friendly

---

## 🧠 Technical Stack

| Component | Selection | Purpose |
|---|---|---|
| Document parsing | PyPDF2 / python-docx / markdown | PDF/DOCX/TXT/MD + image extraction |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Semantic-first chunking (800/150) |
| Text embedding | BAAI/bge-large-zh-v1.5 | Chinese semantic retrieval (bi-encoder) |
| Keyword retrieval | BM25 (rank-bm25) | Literal matching (terms, abbreviations, formula numbers) |
| Fusion | RRF (Reciprocal Rank Fusion) | Merge BM25 + vector results |
| Image embedding | CLIP ViT-B/32 | Cross-modal retrieval (text↔image) |
| Image description | Qwen-VL-Chat | VLM generates text description for LLM |
| Reranker | BAAI/bge-reranker-large | Cross-Encoder precision re-ranking |
| Vector database | Chroma | Persistent storage with metadata |
| Generation | Qwen2.5-7B-Chat (FP16/4bit) | Local LLM |
| Chat format | tokenizer.apply_chat_template | Prevent instruction leakage |
| Evaluation | Custom framework | Recall@K, MRR, Faithfulness, Relevance, Citation |
| Frontend | Gradio | Web interface (text + image input) |

---

## 🏗️ System Pipeline

```text
User Query (text or image)
   ↓
┌──────────────────────────────────────────────┐
│  Retrieval Stage (cascaded)                   │
│                                               │
│  1. Query Enhancement                         │
│     ├─ Query Rewriting (LLM → keywords)       │
│     └─ HyDE (LLM → hypothetical answer)       │
│                                               │
│  2. Multi-path Retrieval                      │
│     ├─ Vector search (bge, original query)    │
│     ├─ Vector search (bge, rewritten query)   │
│     ├─ Vector search (bge, HyDE answer)       │
│     ├─ BM25 search (original query)           │
│     └─ BM25 search (rewritten query)          │
│                                               │
│  3. Fusion & Deduplication                    │
│     └─ RRF (Reciprocal Rank Fusion)           │
│                                               │
│  4. MMR (diversity constraint)                │
│                                               │
│  5. Reranker (Cross-Encoder precision)        │
│     └─ Top-K final results                    │
└──────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────┐
│  Generation Stage                             │
│                                               │
│  6. chat_template construction                │
│  7. Constrained System Prompt + Context       │
│  8. Local LLM generation (Qwen2.5-7B-Chat)    │
│  9. Post-processing                           │
│     ├─ Remove instruction leakage             │
│     ├─ Detect verbatim copying                │
│     └─ Add source references                  │
└──────────────────────────────────────────────┘
   ↓
Answer + Source References + Retrieved Images
```

---

## 📁 Project Structure

```text
rag-academic-qa-agent/
├── README.md
├── requirements.txt
├── main.py                        # 主入口 (CLI / Web / Eval 模式)
├── data/
│   └── eval_dataset.json          # 评估数据集
└── src/
    ├── document_loader.py         # 文档加载 (PDF/DOCX/TXT/MD + 图片提取)
    ├── text_splitter.py           # 语义切分
    ├── image_processor.py         # 多模态: CLIP编码 + VLM图像描述
    ├── vector_store.py            # Chroma 图文混合存储
    ├── bm25_retriever.py          # BM25 关键词检索 + RRF 融合
    ├── reranker.py                # Cross-Encoder 精排
    ├── retriever.py               # 7种检索方法 (similarity/mmr/rewrite/rerank/hybrid/hyde/full)
    ├── rag_chain.py               # RAG链路 (Prompt + 后处理)
    ├── evaluation.py              # 量化评估框架
    ├── llm.py                     # 本地LLM加载
    ├── app.py                     # Gradio前端
    └── utils.py                   # 工具函数
```

---

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download models

```bash
# Embedding (auto-downloads on first run)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"

# Reranker
python -c "from transformers import AutoModel; AutoModel.from_pretrained('BAAI/bge-reranker-large')"

# Generation model (optional)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Chat', local_dir='./models/qwen2.5-7b-chat')"
```

### 3. Put documents in ./docs/

### 4. Run

```bash
# CLI with different retrieval methods
python main.py --model_path ./models/qwen2.5-7b-chat --mode cli --retrieval_method mmr
python main.py --model_path ./models/qwen2.5-7b-chat --mode cli --retrieval_method hybrid
python main.py --model_path ./models/qwen2.5-7b-chat --mode cli --retrieval_method hyde
python main.py --model_path ./models/qwen2.5-7b-chat --mode cli --retrieval_method full

# Web mode
python main.py --model_path ./models/qwen2.5-7b-chat --mode web --retrieval_method full

# Retrieval only (no GPU needed)
python main.py --mode cli --no_model

# Evaluation
python main.py --mode eval --no_model --eval_retrieval_only
python main.py --mode eval --model_path ./models/qwen2.5-7b-chat
```

---

## 📊 7 Retrieval Methods

| Method | Description | Speed | Accuracy |
|---|---|---|---|
| `similarity` | Pure vector similarity | Fastest | Baseline |
| `mmr` | Vector + diversity constraint | Fast | Good |
| `rewrite` | Query rewriting + dual-path merge | Medium | Good |
| `rerank` | Vector + Cross-Encoder re-ranking | Medium | Better |
| `hybrid` | BM25 + vector + RRF + Reranker | Medium | Better |
| `hyde` | HyDE + MMR + Reranker | Slow | Better |
| `full` | Rewrite + HyDE + Hybrid + RRF + Reranker | Slowest | Best |

---

## 📊 Evaluation Framework

| Dimension | Metric | Description |
|---|---|---|
| Retrieval | Recall@1/4/10 | Whether the answer chunk is retrieved |
| Retrieval | MRR | Mean Reciprocal Rank of answer chunk |
| Generation | Faithfulness | Answer grounded in retrieved context |
| Generation | Answer Relevance | Answer contains expected keywords |
| Citation | Citation Accuracy | Source references are correct |
| Safety | Instruction Leakage Rate | System prompt not leaked |
| Safety | Verbatim Copy Rate | No large-scale copying |

```bash
# Compare retrieval methods
python main.py --mode eval --no_model --eval_retrieval_only \
    --eval_methods similarity mmr hybrid hyde full
```

---

## 📊 Key Design Decisions

### 1. Cascaded retrieval (fast → slow, coarse → fine)

Bi-encoder (bge) is fast but coarse (query and doc encoded independently). Cross-Encoder (Reranker) is slow but precise (query and doc encoded together). Use bi-encoder for initial recall, Cross-Encoder for final re-ranking — best of both worlds.

### 2. Hybrid retrieval (BM25 + vector)

Vector search handles semantic matching ("long-tail problem" → "LDAM"). BM25 handles literal matching ("formula 3.2", "ResNet-50"). RRF fuses both without needing score calibration.

### 3. HyDE (Hypothetical Document Embeddings)

User asks a question, LLM generates a hypothetical answer, use that answer (not the question) for retrieval. The answer is closer to the document's semantic space (both are "statements" not "questions").

### 4. chat_template for instruction leakage prevention

Small models (<7B) leak system prompt content when given plain text. Using `tokenizer.apply_chat_template` with proper role tokens eliminates this issue.

### 5. Quantitative evaluation

Every retrieval method can be evaluated with Recall@K, MRR, and generation quality with Faithfulness, Relevance, and Citation Accuracy — turning qualitative claims into quantitative data.

---

## 👤 Author

**Guojie Li**

Master's Student in Information and Communication Engineering
Research interests: Computer Vision, Industrial AI, 3D Reconstruction, RAG Agent, Local LLM Deployment

GitHub: [Aeijou37](https://github.com/Aeijou37)
