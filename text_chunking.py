
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read PDF
reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Divide text into chunks
chunks = splitter.split_text(text)

# Display chunks
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
