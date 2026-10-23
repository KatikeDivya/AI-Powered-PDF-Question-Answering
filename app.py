
import os
import numpy as np
import faiss

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load AI models
model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Store PDF data
chunks = []
index = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():

    global chunks, index

    if "pdf" not in request.files:
        return jsonify({"error": "Please select a PDF."}), 400

    file = request.files["pdf"]

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    try:
        reader = PdfReader(file.stream)

        if reader.is_encrypted:
            return jsonify({"error": "Encrypted PDFs are not supported."}), 400

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            return jsonify({"error": "No readable text found in PDF."}), 400

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        new_chunks = splitter.split_text(text)

        if not new_chunks:
            return jsonify({"error": "Could not create text chunks."}), 400

        embeddings = model.encode(new_chunks)

        embeddings = np.asarray(
            embeddings, dtype="float32"
        )

        new_index = faiss.IndexFlatL2(384)
        new_index.add(embeddings)

        chunks = new_chunks
        index = new_index

        return jsonify({
            "message": "PDF uploaded successfully!",
            "total_chunks": len(chunks)
        })

    except Exception:
        app.logger.exception("PDF processing failed")
        return jsonify({"error": "Unable to process this PDF."}), 500


@app.route("/ask", methods=["POST"])
def ask_question():

    if index is None or not chunks:
        return jsonify({"error": "Please upload a PDF first."}), 400

    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        question_embedding = model.encode([question])

        question_embedding = np.asarray(
            question_embedding, dtype="float32"
        )

        distances, indices = index.search(
            question_embedding, min(3, len(chunks))
        )

        context = "\n\n".join(
            chunks[i] for i in indices[0] if i >= 0
        )

        prompt = f"""
        Answer the user's question using only the PDF context.

        If the answer is not available in the context,
        clearly say it is not available in the PDF.

        PDF Context:
        {context}

        Question:
        {question}

        Answer:
        """

        response = llm.invoke(prompt)

        return jsonify({
            "answer": response.content
        })

    except Exception:
        app.logger.exception("Question answering failed")
        return jsonify({"error": "Unable to generate an answer."}), 500


if __name__ == "__main__":
    app.run(debug=True)
