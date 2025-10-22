import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load environment
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Inisialisasi ulang embedding dan client
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Cek jumlah data yang sudah di-embed
collection_info = client.get_collection(COLLECTION)
print(f"📊 Koleksi aktif: {COLLECTION}")
print(f"🧩 Jumlah vector: {collection_info.points_count}")

# Coba retrieval manual (evaluasi kualitas)
query = input("\n🧠 Masukkan pertanyaan untuk evaluasi: ")
hits = client.search(
    collection_name=COLLECTION,
    query_vector=embeddings.embed_query(query),
    limit=3,
)

print("\n🔍 Top 3 hasil retrieval:")
for i, hit in enumerate(hits, start=1):
    print(f"{i}. Skor: {hit.score:.4f}")
    print(hit.payload.get("text", "")[:250])
    print("-" * 50)
