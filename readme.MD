# 🎓 AI Course Generator — FastAPI Backend

Backend AI untuk platform **LKS Kodeno**. Menerima request dari Laravel backend untuk memproses audio/dokumen dan meng-generate konten pembelajaran berbasis RAG (Retrieval-Augmented Generation).

---

## 🧩 Arsitektur

```
Laravel Backend
      │
      ▼
FastAPI (port 8001)
      │
      ├── Redis Queue ──► Celery Workers
      │                        │
      │                   ┌────┴────────────────┐
      │                   │                     │
      │              Audio Worker          Document Worker
      │           (Faster Whisper)           (Docling)
      │                   │                     │
      │                   └────────┬────────────┘
      │                            ▼
      │                    Gemini Embedding
      │                            │
      │                            ▼
      └──────────────────► Qdrant Vector DB
                                   │
                                   ▼
                            Generate Endpoint
                         (Flashcard / Quiz / Summary / Material)
                                   │
                                   ▼
                            Gemini LLM (RAG)
```

### Komponen Utama

| Komponen                | Fungsi                                              |
| ----------------------- | --------------------------------------------------- |
| 🎙️ **Faster Whisper**   | Speech-to-text dari audio rekaman                   |
| 📄 **Docling**          | Ekstraksi teks struktural dari PDF                  |
| 🔢 **Gemini Embedding** | Generate vektor untuk semantic search               |
| 🗄️ **Qdrant Cloud**     | Vector database untuk penyimpanan & pencarian chunk |
| ✨ **Gemini LLM**       | Generate flashcard, quiz, ringkasan, materi         |
| 🔄 **LangGraph**        | Orchestrasi workflow RAG                            |
| ⚡ **Celery + Redis**   | Async task queue untuk proses berat                 |

---

## ⚙️ Persyaratan

| Tool           | Versi Minimum                  |
| -------------- | ------------------------------ |
| Python         | 3.11+                          |
| Docker         | 24+                            |
| Docker Compose | v2+                            |
| ffmpeg         | (sudah termasuk di Dockerfile) |

Akun eksternal yang dibutuhkan:

- **Gemini API Key** — [aistudio.google.com](https://aistudio.google.com)
- **Qdrant Cloud** — [cloud.qdrant.io](https://cloud.qdrant.io) (cluster gratis tersedia)

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Buat File `.env`

Buat file `.env` di root project berdasarkan template berikut:

```env
# ── App ───────────────────────────────────────────────────────────────
APP_NAME=AI Course Generator
APP_VERSION=1.0.0
DEBUG=false

# ── Gemini ────────────────────────────────────────────────────────────
# Dapatkan dari: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=AIzaSy...
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_GENERATE_MODEL=gemini-2.5-flash

# ── Qdrant Cloud ──────────────────────────────────────────────────────
# Dapatkan dari: https://cloud.qdrant.io → Cluster → API Keys
QDRANT_URL=https://xxxx-xxxx.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
QDRANT_COLLECTION=course_documents
QDRANT_EMBEDDING_DIM=3072

# ── Redis ─────────────────────────────────────────────────────────────
# Tidak perlu diubah jika pakai Docker Compose
REDIS_URL=redis://redis:6379/0

# ── Whisper ───────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_KEEP_IN_MEMORY=false

# ── Upload Limits ─────────────────────────────────────────────────────
MAX_UPLOAD_MB=100
MAX_AUDIO_MB=10
MAX_DOCUMENT_MB=10

# ── CORS ──────────────────────────────────────────────────────────────
ALLOWED_ORIGINS=["http://localhost:8000","http://localhost:3000"]
```

### 3. Jalankan dengan Docker Compose

```bash
docker compose up --build -d
```

Ini akan menjalankan:

- `fastapi` → API server di port **8001**
- `celery_audio` → Worker untuk proses audio
- `celery_document` → Worker untuk proses PDF
- `celery_generate` → Worker untuk generate konten
- `flower` → Celery dashboard di port **5555**
- `redis` → Message broker di port **6379**
- `qdrant` → Vector DB lokal di port **6333** _(opsional, jika tidak pakai Qdrant Cloud)_

### 4. Cek Status

```bash
# Cek semua container berjalan
docker compose ps

# Lihat log FastAPI
docker compose logs -f fastapi

# Lihat log semua worker
docker compose logs -f celery_audio celery_document celery_generate
```

Jika berhasil, buka:

- **API Docs (Swagger):** http://localhost:8001/docs
- **API Docs (ReDoc):** http://localhost:8001/redoc
- **Health Check:** http://localhost:8001/api/v1/health
- **Celery Dashboard:** http://localhost:5555

---

## 🔧 Menjalankan Tanpa Docker (Development)

### 1. Buat Virtual Environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Pastikan `ffmpeg` sudah terinstall di sistem:
>
> ```bash
> # Ubuntu/Debian
> sudo apt install ffmpeg
>
> # Mac (Homebrew)
> brew install ffmpeg
>
> # Windows: download dari https://ffmpeg.org/download.html
> ```

### 3. Jalankan Redis (wajib untuk Celery)

```bash
# Menggunakan Docker
docker run -d -p 6379:6379 redis:7-alpine

# Atau install Redis lokal
```

### 4. Jalankan FastAPI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. Jalankan Celery Workers

Buka terminal terpisah untuk masing-masing worker:

```bash
# Worker audio
celery -A app.workers.celery_app worker -Q audio --loglevel=info --concurrency=2

# Worker document
celery -A app.workers.celery_app worker -Q document --loglevel=info --concurrency=1

# Worker generate
celery -A app.workers.celery_app worker -Q generate --loglevel=info --concurrency=2
```

---

## 📡 API Endpoints

Base URL: `http://localhost:8001/api/v1`

### 🎙️ Audio

| Method | Endpoint                    | Deskripsi                                     |
| ------ | --------------------------- | --------------------------------------------- |
| `POST` | `/audio/transcribe`         | Upload audio → transkripsi → simpan ke Qdrant |
| `GET`  | `/audio/tasks`              | Daftar semua task audio                       |
| `GET`  | `/audio/task/{task_id}`     | Status task tertentu                          |
| `GET`  | `/audio/course/{course_id}` | Task per course                               |

**Contoh upload audio:**

```bash
curl -X POST http://localhost:8001/api/v1/audio/transcribe \
  -F "file=@rekaman.mp3" \
  -F "course_id=course-abc" \
  -F "language=id"
```

### 📄 Document

| Method   | Endpoint                       | Deskripsi                                 |
| -------- | ------------------------------ | ----------------------------------------- |
| `POST`   | `/document/upload`             | Upload PDF → ekstraksi → simpan ke Qdrant |
| `GET`    | `/document/tasks`              | Daftar semua task dokumen                 |
| `GET`    | `/document/task/{task_id}`     | Status task tertentu                      |
| `GET`    | `/document/course/{course_id}` | Task per course                           |
| `DELETE` | `/document/course/{course_id}` | Hapus semua data course dari Qdrant       |

**Contoh upload PDF:**

```bash
curl -X POST http://localhost:8001/api/v1/document/upload \
  -F "file=@materi.pdf" \
  -F "course_id=course-abc"
```

### ✨ Generate

> **Penting:** Request body harus berupa flat JSON — field langsung di root, **bukan** dibungkus `"value"` atau `"summary"`.

| Method | Endpoint               | Deskripsi                         |
| ------ | ---------------------- | --------------------------------- |
| `POST` | `/generate/flashcard`  | Generate flashcard dari materi    |
| `POST` | `/generate/quiz/mc`    | Generate soal pilihan ganda       |
| `POST` | `/generate/quiz/essay` | Generate soal essay               |
| `POST` | `/generate/summary`    | Generate ringkasan materi         |
| `POST` | `/generate/material`   | Generate materi pelajaran lengkap |

**Request body (semua endpoint generate):**

```json
{
  "course_id": "course-abc",
  "source_file": "rekaman.mp3",
  "topic": "pisang",
  "count": 5,
  "difficulty": "medium",
  "language": "id"
}
```

| Field         | Tipe    | Wajib | Deskripsi                                         |
| ------------- | ------- | ----- | ------------------------------------------------- |
| `course_id`   | string  | ❌    | Filter by course. Kosong = global (semua data)    |
| `source_file` | string  | ❌    | Filter by nama file tertentu                      |
| `topic`       | string  | ❌    | Topik spesifik. Kosong = semua materi dalam scope |
| `count`       | integer | ❌    | Jumlah item (default: 5, max: 30)                 |
| `difficulty`  | string  | ❌    | `easy` / `medium` / `hard` — hanya untuk quiz     |
| `language`    | string  | ❌    | `id` (default) atau `en`                          |

**Contoh generate flashcard tentang topik "pisang":**

```bash
curl -X POST http://localhost:8001/api/v1/generate/flashcard \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "pisang",
    "count": 5,
    "language": "id"
  }'
```

**Contoh generate quiz dari course tertentu:**

```bash
curl -X POST http://localhost:8001/api/v1/generate/quiz/mc \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": "course-abc",
    "topic": "fotosintesis",
    "count": 5,
    "difficulty": "medium",
    "language": "id"
  }'
```

### 🔧 Health

```bash
curl http://localhost:8001/api/v1/health
```

---

## 📁 Struktur Project

```
.
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── audio.py       # Endpoint upload audio
│   │       │   ├── document.py    # Endpoint upload PDF
│   │       │   ├── generate.py    # Endpoint generate konten
│   │       │   └── health.py      # Health check
│   │       └── router.py
│   ├── schemas/
│   │   ├── audio.py               # Pydantic models audio
│   │   ├── document.py            # Pydantic models document
│   │   └── generate.py            # Pydantic models generate
│   ├── services/
│   │   ├── langgraph_service.py   # RAG workflow (core logic)
│   │   ├── qdrant_service.py      # Vector DB operations
│   │   ├── embedding_service.py   # Gemini embedding
│   │   ├── llm_service.py         # LLM calls
│   │   ├── whisper_service.py     # Audio transcription
│   │   └── docling_service.py     # PDF extraction
│   ├── workers/
│   │   ├── celery_app.py          # Celery config
│   │   ├── audio_worker.py        # Task: proses audio
│   │   ├── document_worker.py     # Task: proses PDF
│   │   └── generate_worker.py     # Task: generate konten
│   ├── utils/
│   │   ├── chunker.py             # Token-based chunking
│   │   ├── audio_utils.py         # Audio preprocessing
│   │   └── logger.py              # Logger config
│   ├── config.py                  # Settings dari .env
│   └── main.py                    # FastAPI app factory
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                           # ← buat sendiri, jangan di-commit
```

---

## 🔍 Troubleshooting

**Error 403 Forbidden dari Qdrant:**

```
{"detail": "Unexpected Response: 403 (Forbidden)"}
```

→ `QDRANT_API_KEY` di `.env` salah atau terpotong. Salin ulang dari Qdrant Cloud dashboard (API Keys).

**Topic tidak relevan di hasil generate:**

Pastikan request body dikirim sebagai flat JSON langsung, **bukan** dibungkus `value`:

```json
// ✅ BENAR
{"topic": "pisang", "count": 5}

// ❌ SALAH (format Swagger UI lama)
{"summary": "...", "value": {"topic": "pisang", "count": 5}}
```

**Celery worker tidak berjalan:**

```bash
# Cek log worker
docker compose logs celery_audio

# Restart worker
docker compose restart celery_audio celery_document celery_generate
```

**Port 8001 sudah dipakai:**

```bash
# Ganti port di docker-compose.yml
ports:
  - "8002:8000"   # ubah 8001 ke port lain
```

---

## 📝 Lisensi

MIT License — bebas digunakan dan dimodifikasi.
