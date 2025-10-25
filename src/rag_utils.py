import os
from langchain_core.documents import Document

def simple_chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    chunks = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(type(doc)(page_content=chunk_text, metadata=getattr(doc, 'metadata', {})))
            start += chunk_size - chunk_overlap
    return chunks
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader

PDF_PATH = os.path.join(os.path.dirname(__file__), "dmv.pdf")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "dmv_index.faiss")


def build_or_load_index():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    chunks = simple_chunk_documents(docs, chunk_size=1000, chunk_overlap=200)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment. Please add it to your .env.local or environment variables.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if os.path.exists(INDEX_PATH):
        db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_PATH)
    return db


def query_index(query, k=3):
    db = build_or_load_index()
    results = db.similarity_search(query, k=k)
    return results
