from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from langchain.docstore.document import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from app.rag_chain import build_chain
import os


app = FastAPI(title="RAG – Qdrant + LangChain")


class AskReq(BaseModel):
question: str


class AskResp(BaseModel):
answer: str
sources: List[str]


class UpdateReq(BaseModel):
texts: List[str]
labels: List[str] | None = None


@app.on_event("startup")
async def _startup():
global chain
chain = build_chain()


@app.post("/ask", response_model=AskResp)
async def ask(req: AskReq):
result = chain({"query": req.question})
answer = result["result"]
srcs = []
for d in result.get("source_documents", []):
srcs.append(str(d.metadata.get("source", d.metadata.get("title", "unknown"))))
# unikkan sumber
srcs = list(dict.fromkeys(srcs))
return AskResp(answer=answer, sources=srcs)


@app.post("/update")
async def update(req: UpdateReq):
# tambahkan dokumen baru ke Qdrant
embeddings = SentenceTransformerEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY") or None)
vectordb = Qdrant(client=client, collection_name=os.getenv("QDRANT_COLLECTION", "rag_documents"), embeddings=embeddings)


docs = []
for i, t in enumerate(req.texts):
label = req.labels[i] if req.labels and i < len(req.labels) else f"api_added_{i}.txt"
docs.append(Document(page_content=t, metadata={"source": f"api::{label}"}))


vectordb.add_documents(docs)
return {"added": len(docs)}