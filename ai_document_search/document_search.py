# Simple AI-powered document search and summarization

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)

# -------------------------------
# Utility functions
# -------------------------------

def extract_text(file_path: Path) -> str:
    """Extract text from PDF, docx or txt."""
    if file_path.suffix.lower() == '.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file_path.suffix.lower() in {'.docx', '.doc'}:
        from docx import Document
        doc = Document(str(file_path))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        text = file_path.read_text(encoding='utf-8', errors='ignore')
    return text


def split_into_chunks(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into roughly ``chunk_size`` word chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    count = 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.extend(words)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_index(doc_dir: Path, index_path: Path):
    """Ingest documents from ``doc_dir`` and build search index at ``index_path``."""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = []
    metadata = []
    for file in doc_dir.iterdir():
        if not file.suffix.lower() in {'.pdf', '.docx', '.doc', '.txt'}:
            continue
        text = extract_text(file)
        for i, chunk in enumerate(split_into_chunks(text)):
            emb = model.encode(chunk)
            embeddings.append(emb)
            metadata.append({'document': file.name, 'chunk': i, 'text': chunk})

    os.makedirs(index_path, exist_ok=True)
    with open(index_path / 'embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    with open(index_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f)


def load_index(index_path: Path):
    with open(index_path / 'embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    with open(index_path / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return embeddings, metadata


def search(query: str, embeddings: List, metadata: List[Dict], top_k: int = 3):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_emb = model.encode(query)
    import numpy as np
    emb_matrix = np.vstack(embeddings)
    norms = np.linalg.norm(emb_matrix, axis=1) * (np.linalg.norm(query_emb) + 1e-10)
    sims = emb_matrix @ query_emb / norms
    top_idx = sims.argsort()[-top_k:][::-1]
    return [metadata[i] for i in top_idx]


def summarize(text: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    summary_ids = model.generate(**inputs, max_length=150)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description='Document search and summarize')
    subparsers = parser.add_subparsers(dest='command', required=True)

    ingest_p = subparsers.add_parser('ingest', help='Build an index from documents')
    ingest_p.add_argument('document_dir', type=Path)
    ingest_p.add_argument('index_dir', type=Path)

    query_p = subparsers.add_parser('query', help='Search documents')
    query_p.add_argument('index_dir', type=Path)
    query_p.add_argument('query', type=str)
    query_p.add_argument('--top_k', type=int, default=3)

    args = parser.parse_args()

    if args.command == 'ingest':
        build_index(args.document_dir, args.index_dir)
    elif args.command == 'query':
        embeddings, metadata = load_index(args.index_dir)
        results = search(args.query, embeddings, metadata, args.top_k)
        combined = "\n".join(r['text'] for r in results)
        print("Top results:\n")
        for r in results:
            print(f"{r['document']} [chunk {r['chunk']}]\n")
        summary = summarize(combined)
        print("\nSummary:\n", summary)


if __name__ == '__main__':
    main()
