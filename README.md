# 🇮🇩 Mini Project: Retrieval Augmented Generation (RAG)

## Deskripsi
Proyek ini mengimplementasikan **RAG Chatbot** berbasis teks sejarah kemerdekaan Indonesia.
Sistem mengambil pengetahuan dari dokumen lokal (PDF/TXT) dan public API (Wikipedia), 
menggunakan **LangChain**, **Qdrant**, dan **OpenAI GPT-4o-mini**.

---

## Arsitektur Sistem
1. **Ingest Data** – Ekstraksi teks & vectorization (`ingest.py`)
2. **Store to Qdrant** – Simpan embedding ke Vector DB
3. **Retrieval** – Cari vektor relevan saat user bertanya
4. **LLM Response** – Hasil dikembalikan lewat OpenAI LLM

---

# Cara Menjalankan
### 1️⃣ Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
