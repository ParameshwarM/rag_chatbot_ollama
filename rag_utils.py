import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

# Load components
def load_rag_components():
    index = faiss.read_index("faiss_index.idx")
    with open("chunks.pkl", "rb") as f: chunks = pickle.load(f)
    with open("metadata.pkl", "rb") as f: metadata = pickle.load(f)
    embedder = SentenceTransformer("embedding_model")
    return index, chunks, metadata, embedder

# Retrieve top-k chunks
def retrieve_chunks(query, embedder, index, chunks, top_k=5, threshold=0.0):
    q_embed = embedder.encode([query])
    D, I = index.search(np.array(q_embed), top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < len(chunks):
            # No threshold filter — always append
            results.append(chunks[idx])
    print(results)
    return results



def test_retrieval_only(query, top_k=5):
    print("entered test retrieval")
    index, chunks, metadata, embedder = load_rag_components()
    top_chunks = retrieve_chunks(query, embedder, index, chunks, top_k=top_k)
    for i, chunk in enumerate(top_chunks):
        print(f"\n--- CHUNK {i+1} ---\n{chunk[:500]}\n")

print("Hiiiiiiiiiii")
test_retrieval_only("What is covered under the Bronze plan?")


# Run LLaMA via Ollama
def generate_answer_llama(context, question):
    prompt = f"""Use the context below to answer the question. 
If the answer is not in the context, respond with "I don't know".

Context:
{context}

Question:
{question}
"""
    print("\n🔍 Prompt Sent to LLaMA:\n" + prompt[:2000])  # show only first 2000 chars
    result = subprocess.run(
        ["ollama", "run", "llama3", "--", prompt],
        capture_output=True,
        text=True
    )
    print("\n🔁 LLaMA Response:\n", result.stdout.strip())
    return result.stdout.strip()
