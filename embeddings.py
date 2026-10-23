
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text chunks
chunks = [
    "Python is a programming language.",
    "Python is used for machine learning.",
    "Java is another programming language."
]

# Convert text into embeddings
embeddings = model.encode(chunks)

# Display embeddings
print("Embeddings:")
print(embeddings)

# Display embedding dimensions
print("\nEmbedding shape:", embeddings.shape)
