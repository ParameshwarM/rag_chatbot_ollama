# rag_chatbot_ollama


# 🛡️ Insurance RAG Chatbot – Installation & Hosting Guide

This guide walks you through setting up, running, and optionally hosting the RAG-based Insurance Chatbot using LLaMA 3.2 via Ollama + Flask + FAISS.

---

## ✅ Features
- 🔎 Retrieval-Augmented Generation using FAISS
- 🧠 Powered by LLaMA 3.2 (Ollama)
- 💬 Clean UI with Chat Bubbles + Dark Mode
- 📄 Document-based answers only
- ❓ Says "I don't know" if out-of-context

---

## 🧰 Prerequisites

- Python 3.10 or above
- Ollama installed locally
- Git (for version control)
- `llama3` model pulled via Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLaMA 3.2
ollama pull llama3
```

---

## 📦 Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Document Embeddings

Place your `.pdf` and `.docx` files inside a `docs/` folder, then run:

```bash
python build_index.py
```

This will generate:
- `faiss_index.idx`
- `chunks.pkl`
- `metadata.pkl`
- `embedding_model/`

---

### 4. Start Ollama in a Terminal

```bash
ollama run llama3
```

> Keep this running in one terminal window.

---

### 5. Start the Flask App

In another terminal:

```bash
python app.py
```

Then open: [http://localhost:5000](http://localhost:5000)


---

## 🛠 Project Structure

```
rag-chatbot/
├── app.py
├── build_index.py
├── rag_utils.py
├── requirements.txt
├── faiss_index.idx
├── chunks.pkl
├── metadata.pkl
├── embedding_model/
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── docs/
    └── *.pdf / *.docx
```

---

## 🧪 Sample Questions to Try

- What is the deductible in the Bronze 5000 plan?
- Is preventive care covered in the HSA plan?
- What medical questions are required to apply?
- Does the plan include emergency room coverage?

---
