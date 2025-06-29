import os, re, pickle
import fitz  
from docx import Document
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_PATH = "C:/projects/rag_chatbot_ollama/docs"  
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        all_text = ""
        for page in doc:
            page_text = page.get_text()
            all_text += page_text
        return all_text
    except Exception as e:
        print(f"Failed to extract from PDF: {path}, error: {e}")
        return ""


def extract_text_from_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Failed to extract from DOCX: {path}, error: {e}")
        return ""


def chunk_text(text, chunk_size=500, overlap=100):
    text = re.sub(r'\s+', ' ', text.strip())
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def load_documents(folder):
    chunks, metadata = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        print(f"Processing file: {file}")
        if file.endswith(".pdf"):
            content = extract_text_from_pdf(path)
        elif file.endswith(".docx"):
            content = extract_text_from_docx(path)
        else:
            print(f"Skipping unsupported file: {file}")
            continue

        print(f"Extracted text length: {len(content)}")
        doc_chunks = chunk_text(content)
        print(f"Chunks created: {len(doc_chunks)}")

        for chunk in doc_chunks:
            chunks.append(chunk)
            metadata.append({"source": file})
    return chunks, metadata


def main():
    chunks, metadata = load_documents(DOCS_PATH)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save outputs
    faiss.write_index(index, "faiss_index.idx")
    with open("chunks.pkl", "wb") as f: pickle.dump(chunks, f)
    with open("metadata.pkl", "wb") as f: pickle.dump(metadata, f)
    model.save("embedding_model")

    print("✅ FAISS index and embedding model saved.")

if __name__ == "__main__":
    main()
