
import os
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Groq API key not found. Check your .env file.")

# 1. Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

if not text.strip():
    raise ValueError("No readable text found in PDF.")

print("PDF loaded successfully!")

# 2. Split PDF text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings generated!")

# 5. Create FAISS database
index = faiss.IndexFlatL2(384)

index.add(embeddings)

print("FAISS database created!")

# 6. Connect to Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)

print("\nPDF Question Answering System is Ready!")

# 7. Ask questions
while True:

    question = input("\nAsk a question (type exit to stop): ")

    if question.lower() == "exit":
        print("Thank you for using the PDF chatbot!")
        break

    # Convert question into embedding
    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Retrieve relevant chunks
    distances, indices = index.search(
        question_embedding,
        min(3, len(chunks))
    )

    context = "\n\n".join(
        chunks[i] for i in indices[0] if i >= 0
    )

    # 8. Send context and question to LLM
    prompt = f"""
    You are a PDF question answering assistant.

    Answer the question using only the information
    provided in the PDF context below.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the uploaded PDF.

    PDF Context:
    {context}

    User Question:
    {question}

    Answer:
    """

    # 9. Generate AI answer
    response = llm.invoke(prompt)

    print("\nAI Answer:")
    print(response.content)
