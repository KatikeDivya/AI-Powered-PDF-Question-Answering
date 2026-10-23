
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample document chunks
chunks = [
    "Python is a popular programming language.",
    "Python is used for machine learning and AI.",
    "Java is an object-oriented programming language.",
    "Machine learning helps computers learn from data."
]

# Convert chunks into embeddings
embeddings = model.encode(chunks)

# Convert embeddings into NumPy array
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(384)

# Store embeddings in FAISS
index.add(embeddings)

# Ask a question
question = "What is Python used for?"

# Convert question into embedding
question_embedding = model.encode([question])

question_embedding = np.array(
    question_embedding
).astype("float32")

# Search for the 2 most similar chunks
distances, indices = index.search(question_embedding, 2)

# Display results
print("\nQuestion:", question)

print("\nRelevant information:")

for i in indices[0]:
    print(chunks[i])
