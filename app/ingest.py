import os
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, WikipediaLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from app.utils import clean_text

load_dotenv()

RAW_DIR = "data/raw"
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

TOPICS = [
    "Sejarah Indonesia",
    "Proklamasi Kemerdekaan Indonesia",
    "Pendudukan Jepang di Indonesia",
    "NICA",
]


from langchain_community.document_loaders import WikipediaLoader
def load_wikipedia_docs():
    topics = ["Proklamasi Kemerdekaan Indonesia", "NICA Indonesia"]
    docs = []
    for t in topics:
        docs.extend(WikipediaLoader(query=t, lang="id").load())
    return docs


def load_local_docs() -> List:
    """Membaca file teks atau PDF dari folder data/raw"""
    docs = []
    if not os.path.isdir(RAW_DIR):
        return docs
    for fname in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, fname)
        if fname.lower().endswith((".txt", ".md")):
            docs += TextLoader(path, encoding="utf-8").load()
        elif fname.lower().endswith(".pdf"):
            docs += PyMuPDFLoader(path).load()
    for d in docs:
        d.page_content = clean_text(d.page_content)
    return docs


def load_wikipedia_docs() -> List:
    """Mengambil artikel dari Wikipedia sesuai topik"""
    docs = []
    for topic in TOPICS:
        w = WikipediaLoader(query=topic, load_max_docs=2, lang="id")
        for d in w.load():
            d.page_content = clean_text(d.page_content)
            docs.append(d)
    return docs


if __name__ == "__main__":
    # 1️⃣ Siapkan splitter & embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

  
    docs = load_local_docs() + load_wikipedia_docs()
    if not docs:
        raise SystemExit("❌ Tidak ada dokumen untuk di-embed. Tambahkan file di data/raw atau aktifkan TOPICS Wikipedia.")

    chunks = splitter.split_documents(docs)

  
  
from langchain_community.vectorstores import Qdrant

vs = Qdrant.from_documents(
    documents=chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION,
)

print(f"✅ Sukses ingest {len(chunks)} chunks ke koleksi '{COLLECTION}' di Qdrant!")
