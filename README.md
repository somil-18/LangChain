# LangChain Components & LLM Pipeline Modules

A modular collection of LangChain components designed for building real-world LLM pipelines such as RAG systems, document-Q&A tools, vector search engines, and agent-based workflows.  
The repository follows a clean, scalable structure similar to production-ready LLM projects.

---

## 📁 Repository Structure

### **Chains/**
Implementations of Chains, Sequential Chains, Parallel Chains, Branch Chains, Runnables and multi-step reasoning pipelines.

### **DocumentLoader/**
Loaders for PDFs, text files, web pages, and other formats using LangChain’s document loader ecosystem.

### **Models/**
Configuration of OpenAI, Groq, and local LLMs, optimized for different inference use cases.

### **Outputs/**
Saved outputs used for evaluation and debugging.  
Includes:
- Models using **built-in Output Parsers**  
- Models **without** output parsers (manual parsing logic)

### **Prompts/**
A collection of:
- **Dynamic prompts**
- **PromptTemplate**
- **ChatPromptTemplate**
- Structured prompts for classification, extraction, summarization, and agent workflows.

### **TextSplitters/**
Text chunking techniques including:
- **Character-based splitting**  
- **Text-structure-based splitting**  
- **Markdown-aware splitting**  
- **Semantic search splitters** (embedding-aware chunking)

### **VectorDatabase/**
FAISS, Chroma, and other vector store experiments for indexing, embedding storage, and optimized retrieval.

### **Retrievers/**
High-quality retriever implementations, including:
- **Similarity Search Retriever**
- **MMR (Max Marginal Relevance) Retriever**
- **Context-compressed Retriever**
- **Multi-Query Retriever**
- **Wikipedia Retriever**

---

## 🧩 What This Repo Demonstrates
- Modular LLM pipeline architecture  
- Clean component separation (prompts, models, splitters, retrievers, vector DB)  
- Real patterns used in RAG systems and agent workflows  
- Reusable building blocks for rapid prototype development  

---

## 🛠 Tech Stack
- Python  
- LangChain  
- Gemini/ HF 
- FAISS, Chroma  
- Embeddings, Retrieval pipelines  

---

## 🚀 Ongoing Work
I’m actively learning and will keep adding new modules, improvements, and experiments as I progress.

---

## 📌 Note
Actively expanding this repository — more modules and advanced examples will be added as I continue improving my LangChain and LLM application skills.
