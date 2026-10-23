# AI-Powered PDF Question Answering System using RAG

## Project Overview

The AI-Powered PDF Question Answering System is a web application that allows users to upload PDF documents and ask questions based on their content.

It uses **Retrieval Augmented Generation (RAG)** to retrieve relevant information from uploaded documents and generate answers using a Large Language Model (LLM).

The application provides an interactive interface built using Flask, HTML and CSS.

## Features

- Upload PDF documents.
- Extract text from PDF files.
- Split extracted text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Store and search embeddings using FAISS.
- Retrieve relevant information based on user questions.
- Generate answers using the GPT-OSS-20B model through Groq API.
- Interactive chatbot interface.
- Answers are generated using the uploaded document's content.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Flask | Web application framework |
| HTML & CSS | Frontend design |
| LangChain | Text splitting and RAG workflow |
| PyPDF | Extracting text from PDF files |
| Sentence Transformers | Generating text embeddings |
| FAISS | Similarity search and retrieval |
| Groq API | Accessing the language model |
| GPT-OSS-20B | Generating AI answers |
| NumPy | Numerical operations |

## How It Works

The project follows the Retrieval Augmented Generation (RAG) architecture.

1. **PDF Upload:** The user uploads a PDF document through the web interface.
2. **Text Extraction:** PyPDF extracts text from the uploaded document.
3. **Text Chunking:** LangChain splits the extracted text into smaller chunks.
4. **Embedding Generation:** Sentence Transformers converts text chunks into numerical vectors.
5. **Vector Storage:** FAISS stores the vectors and enables similarity search.
6. **Question Processing:** The user enters a question about the PDF.
7. **Information Retrieval:** FAISS retrieves the most relevant text chunks.
8. **Answer Generation:** The retrieved context and question are sent to GPT-OSS-20B through the Groq API.
9. **Display Answer:** The generated answer is displayed in the Flask web application.

## Project Structure

```text
AI-Powered-PDF-Question-Answering/
│
├── app.py
├── pdf_qa.py
├── pdf_reader.py
├── text_chunking.py
├── embeddings.py
├── faiss_search.py
├── pdf_vector_search.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

**Note:** The `.env` file is used locally to store the Groq API key. It must not be uploaded to GitHub.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd AI-Powered-PDF-Question-Answering
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Groq API Key

Create a `.env` file in the project folder and add:

```text
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

Never share or upload your API key.

### 7. Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

Upload a PDF and start asking questions!

## RAG Architecture

```text
        PDF Upload
             |
             v
       Text Extraction
             |
             v
        Text Chunking
             |
             v
    Embedding Generation
             |
             v
       FAISS Vector Index
             |
             v
       User Question
             |
             v
      Similarity Search
             |
             v
     Relevant Text Chunks
             |
             v
      Groq API - GPT-OSS-20B
             |
             v
       Generated Answer
             |
             v
        Flask Web UI
```

