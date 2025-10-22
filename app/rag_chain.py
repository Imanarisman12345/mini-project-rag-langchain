import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

# 🔹 Muat file .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

# 1️⃣ Hubungkan client Qdrant (REST API)
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# 2️⃣ Gunakan koleksi yang sudah ada
db = Qdrant(
    client=client,
    collection_name=COLLECTION,
    embedding_function=embeddings,
)

# 3️⃣ Buat retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 4️⃣ Integrasi LLM (OpenAI)
llm = OpenAI(temperature=0.2, api_key=OPENAI_API_KEY)

# 5️⃣ Buat chain RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
)

# 6️⃣ Jalankan chatbot interaktif
if __name__ == "__main__":
    print("\n🤖 RAG Chatbot siap digunakan! Ketik 'exit' untuk keluar.")
    while True:
        question = input("\n🧠 Masukkan pertanyaan: ")
        if question.lower() == "exit":
            print("👋 Terima kasih, sampai jumpa!")
            break
        answer = qa_chain.run(question)
        print(f"\n💬 Jawaban: {answer}")
