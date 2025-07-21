# RAGBot: Retrieval-Augmented Generation for IT Support Tickets

Welcome to **RAGBot** — a full-stack chatbot powered by **vector search**, **FAISS**, and a **quantized open-source LLM (Zephyr-7B-Beta)**. It provides **contextual answers** to IT support queries using historical ticket data.

Built entirely in **Google Colab**, optimized for **free-tier infrastructure**, and designed to be fully modular and extensible.

---

## 📌 Project Overview

| 🔧 Module        | 🔍 Description                                                                 |
|------------------|--------------------------------------------------------------------------------|
| 🔍 Retrieval     | FAISS + SentenceTransformer embeddings to fetch relevant ticket responses      |
| 🧠 Generation     | Zephyr-7B-Beta (4-bit quantized) model for empathetic, LLM-generated responses |
| 📊 Evaluation     | Precision@k metric to evaluate retrieval quality                              |
| 💬 Interface      | Command-line or Python-callable function for RAG-based Q&A                   |

---

We use TheBloke/zephyr-7B-beta-GGUF in 4-bit quantized format for efficiency on Colab.

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(
          "TheBloke/zephyr-7B-beta-GGUF", 
            use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/zephyr-7B-beta-GGUF",
    device_map="auto",
    load_in_4bit=True
)

🛠️ TODOs & Improvements
	•	🔁 Switch to all-mpnet-base-v2 or e5-large-v2 embeddings
	•	🔍 Add BM25 + dense hybrid retrieval
	•	🧪 Implement BLEU/ROUGE/LlamaIndex eval for generated answers
	•	🌍 Wrap with Gradio for interactive UI
	•	☁️ Deploy backend to AWS EC2 free tier
