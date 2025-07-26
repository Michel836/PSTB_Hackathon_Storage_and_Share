# AI-Powered Document Search and Summarization

This folder contains a small proof-of-concept tool that indexes documents and performs semantic search with summarisation. It is designed to work on CPUs with minimal dependencies.

## Requirements

- Python 3.12
- Packages installed in this repository (`sentence-transformers`, `transformers`, `PyPDF2`, `python-docx`, `nltk`)

## Usage

1. **Ingest Documents**

   Provide a directory containing PDF, Word or text files and build an index:

   ```bash
   python document_search.py ingest <documents_dir> <index_dir>
   ```

   This extracts text, splits it into chunks, generates embeddings and saves them in `<index_dir>`.

2. **Search and Summarize**

   Query the index to retrieve relevant chunks and get a short summary:

   ```bash
   python document_search.py query <index_dir> "your question here" --top_k 3
   ```

   The script prints which document chunks were most relevant and outputs a short summary generated with the `t5-small` model.

## Notes

This implementation keeps embeddings in plain pickle files and uses a simple cosine similarity search with NumPy. For larger collections you may replace this logic with a dedicated vector database such as FAISS or Pinecone.
