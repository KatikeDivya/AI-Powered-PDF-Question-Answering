
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# 1. Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

print("PDF text extracted successfully!")

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

if not chunks:
    raise ValueError("No text found in PDF.")

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert chunks into embeddings
embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings generated!")

# 5. Create FAISS index
index = faiss.IndexFlatL2(384)

index.add(embeddings)

print("Embeddings stored in FAISS!")

# 6. Ask a question
question = input("\nAsk a question about your PDF: ")

# 7. Convert question into embedding
question_embedding = model.encode([question])

question_embedding = np.array(
    question_embedding
).astype("float32")

# 8. Search relevant chunks
distances, indices = index.search(
    question_embedding,
    min(3, len(chunks))
)

# 9. Display results
print("\nRelevant information from PDF:")

for i in indices[0]:
    print("\n", chunks[i])
