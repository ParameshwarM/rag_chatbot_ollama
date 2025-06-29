from flask import Flask, render_template, request
from rag_utils import load_rag_components, retrieve_chunks, generate_answer_llama

app = Flask(__name__)

# Load components once
index, chunks, metadata, embedder = load_rag_components()

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = ""
    question = ""
    if request.method == "POST":
        question = request.form["question"]
        top_chunks = retrieve_chunks(question, embedder, index, chunks)
        if top_chunks:
            context = "\n\n".join(top_chunks)
            answer = generate_answer_llama(context, question)
        else:
            answer = "I don't know"
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)