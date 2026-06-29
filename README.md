# RAG Academic QA Agent

A local academic document question-answering agent based on **Retrieval-Augmented Generation (RAG)**, **LangChain**, **Chroma**, and local LLM deployment.

This project aims to build a lightweight and extensible academic knowledge base system that can parse local documents, construct vector databases, retrieve relevant contexts, and generate grounded answers with source tracing.

---

## 📌 Project Overview

With the rapid growth of academic papers, technical reports, and research notes, it becomes increasingly difficult to efficiently search, understand, and summarize large collections of documents.

This project explores the use of RAG technology to build a local academic knowledge base QA system. The system supports document parsing, text chunking, vector embedding, persistent vector storage, semantic retrieval, and local question answering.

The main goal is to enable users to ask questions over their own academic documents and receive answers grounded in the uploaded materials.

---

## ✨ Key Features

* Local academic document question answering
* Support for multiple document formats
* Document parsing and text chunking
* Vector database construction with Chroma
* Retrieval-Augmented Generation pipeline
* MMR-based retrieval for better context diversity
* Multi-document management and cross-document retrieval
* Query rewriting for improved retrieval quality
* Source tracing for answer verification
* Prompt engineering for more stable responses
* Local LLM deployment support

---

## 🧠 Technical Stack

### Programming Language

* Python

### Frameworks and Libraries

* LangChain
* Chroma
* PyTorch
* Transformers
* Sentence Transformers

### Core Techniques

* Retrieval-Augmented Generation
* Vector Embedding
* Semantic Search
* MMR Retrieval
* Query Rewriting
* Prompt Engineering
* Local Large Language Model Deployment

---

## 🏗️ System Pipeline

```text
Documents
   |
   v
Document Parsing
   |
   v
Text Chunking
   |
   v
Embedding Generation
   |
   v
Chroma Vector Database
   |
   v
Semantic Retrieval / MMR Retrieval
   |
   v
Prompt Construction
   |
   v
Local LLM Generation
   |
   v
Answer with Source References
```

---

## 📂 Supported Document Types

The system is designed to support common academic document formats, including:

* PDF
* DOCX
* TXT
* Markdown

---

## 🚀 Main Modules

### 1. Document Loader

Parses different types of documents and extracts clean text content.

### 2. Text Splitter

Splits long documents into smaller chunks for efficient retrieval and embedding.

### 3. Vector Store

Uses Chroma to store document embeddings persistently.

### 4. Retriever

Retrieves relevant document chunks based on user queries.

### 5. RAG Chain

Combines retrieved contexts with user queries and generates answers through a local language model.

### 6. Source Tracing

Returns source information to help users verify the generated answers.

---

## 📊 Project Highlights

* Built a complete local RAG pipeline from document parsing to answer generation.
* Implemented multi-document retrieval and cross-document question answering.
* Improved retrieval quality using MMR retrieval and query rewriting.
* Designed constrained system prompts to reduce hallucination and improve answer format.
* Supported local model deployment for privacy-friendly academic QA.

---

## 🔧 Installation

```bash
git clone https://github.com/Aeijou37/rag-academic-qa-agent.git
cd rag-academic-qa-agent
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main.py
```

Example questions:

```text
What is the main contribution of this paper?

What datasets are used in the experiment?

Summarize the method section of the uploaded document.

Compare the differences between these two papers.
```

---

## 📁 Project Structure

```text
rag-academic-qa-agent/
├── README.md
├── requirements.txt
├── main.py
├── configs/
├── data/
├── docs/
├── examples/
└── src/
    ├── document_loader.py
    ├── text_splitter.py
    ├── vector_store.py
    ├── retriever.py
    ├── rag_chain.py
    └── utils.py
```

---

## 📌 Current Status

This repository is currently being organized and will be continuously updated.

Planned updates:

* Add complete source code
* Add example documents
* Add configuration files
* Add local LLM deployment instructions
* Add screenshots and demo examples
* Add evaluation and ablation notes

---

## 👤 Author

**Guojie Li**

Master's Student in Information and Communication Engineering
Research interests: Computer Vision, Industrial AI, 3D Reconstruction, RAG Agent, and Local LLM Deployment

GitHub: [Aeijou37](https://github.com/Aeijou37)
