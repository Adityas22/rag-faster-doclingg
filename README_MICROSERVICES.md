# AI Course Generator — Microservices Architecture (v2)

## 🏗️ Arsitektur

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                            │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────────┐  │
│  │  Laravel │───▶│  FastAPI │───▶│  Redis (Celery Broker)│  │
│  │ (client) │    │ :8000    │    │  :6379               │  │
│  └──────────┘    └──────────┘    └───────────────────────┘  │
│                                         │                    │
│                         ┌───────────────┼───────────────┐   │
│                         ▼               ▼               ▼   │
│                  ┌─────────────┐ ┌──────────────┐ ┌──────┐  │
│                  │  Docling    │ │   Whisper    │ │ Gen  │  │
│                  │  Worker     │ │   Worker     │ │Worker│  │
│                  │  (Service A)│ │  (Service B) │ │      │  │
│                  │  PyTorch    │ │  PyTorch     │ │      │  │
│                  │  (isolated) │ │  (isolated)  │ │      │  │
│                  └─────────────┘ └──────────────┘ └──────┘  │
│                         │               │               │   │
│                         └───────────────┼───────────────┘   │
│                                         ▼                   │
│                              ┌───────────────────┐          │
│                              │  Qdrant :6333     │          │
│                              │  (Vector DB)      │          │
│                              └───────────────────┘          │
│                                                              │
│              ┌───────────────────────────────────┐          │
│              │  Ollama :11434 (Llama LLM)        │          │
│              │  PRIMARY generator, no quota limit│          │
│              └───────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Masalah yang Diselesaikan

| Masalah | Solusi |
|---|---|
| `c10.dll` conflict (Windows) | Setiap worker punya container Linux sendiri |
| API timeout saat Docling berjalan | Celery async task, FastAPI hanya kirim task_id |
| Kuota Gemini habis untuk generate | Llama via Ollama sebagai primary LLM |
| Docling OCR tidak aktif | `do_ocr=True` + `do_table_structure=True` di Docling Worker |

## 🚀 Cara Menjalankan

### 1. Prerequisites
```bash
# Install Docker & Docker Compose
docker --version   # >= 20.10
docker compose version  # >= 2.0
```

### 2. Setup Environment
```bash
# Edit .env, isi API keys
cp .env .env.backup
nano .env

# Wajib diisi:
# GEMINI_API_KEY=...         (untuk embedding)
# QDRANT_URL=...             (Qdrant Cloud atau lokal)
# QDRANT_API_KEY=...         (jika pakai Qdrant Cloud)
```

### 3. Jalankan Semua Services
```bash
docker compose up -d

# Cek status
docker compose ps

# Lihat log semua service
docker compose logs -f

# Lihat log service tertentu
docker compose logs -f docling_worker
docker compose logs -f whisper_worker
```

### 4. Verifikasi
```bash
# Health check
curl http://localhost:8000/api/v1/health

# API Docs
open http://localhost:8000/docs

# Celery Flower Dashboard (monitoring tasks)
open http://localhost:5555
# Login: admin / admin123
```

## 📦 Struktur Dockerfile

| Dockerfile | Service | Konten |
|---|---|---|
| `Dockerfile.api` | fastapi, celery_generate, flower | FastAPI + Celery (ringan, tanpa PyTorch berat) |
| `Dockerfile.docling` | docling_worker | Docling + OCR + PyMuPDF |
| `Dockerfile.whisper` | whisper_worker | Faster-Whisper + Audio libs |

## 🦙 Llama / Ollama

Ollama berjalan otomatis dan pull model `llama3.2` saat startup.

### Ganti Model
```bash
# Edit docker-compose.yml bagian ollama entrypoint:
# ollama pull llama3.2  →  ollama pull llama3.1  (lebih pintar, lebih berat)
# atau:
# ollama pull mistral   (ringan + cepat)
# ollama pull phi3      (sangat ringan)
# ollama pull qwen2.5   (bagus untuk bahasa Indonesia)

# Atau pull manual setelah container jalan:
docker exec -it ai_course_ollama ollama pull qwen2.5
```

### GPU Support
Uncomment bagian `deploy.resources` di `docker-compose.yml` untuk service `ollama`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔑 Environment Variables Penting

| Variable | Default | Keterangan |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Model Llama yang dipakai |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL Ollama (nama service Docker) |
| `DOCLING_OCR_ENABLED` | `true` | Aktifkan OCR di Docling Worker |
| `DOCLING_TABLE_STRUCTURE` | `true` | Aktifkan table parsing Docling |
| `WHISPER_KEEP_IN_MEMORY` | `true` (di whisper container) | Cache model Whisper di memori |
| `GEMINI_API_KEY` | - | Wajib untuk embedding |
| `QDRANT_URL` | - | Wajib untuk vector storage |

## 🛑 Troubleshooting

```bash
# Restart service tertentu
docker compose restart docling_worker

# Rebuild image setelah update kode
docker compose build docling_worker
docker compose up -d docling_worker

# Lihat penggunaan resource
docker stats

# Masuk ke container untuk debug
docker exec -it ai_course_docling_worker bash

# Hapus cache Docling (jika ingin re-parse semua PDF)
docker exec ai_course_docling_worker find /app/temp -name "*_docling_cache.json" -delete

# Reset semua (HATI-HATI: hapus semua data volumes)
docker compose down -v
```

## 📊 Flower — Celery Task Monitoring

Akses dashboard di http://localhost:5555 (login: `admin` / `admin123`)

Tampilkan:
- Task yang sedang berjalan (STARTED)
- Task yang berhasil (SUCCESS)
- Task yang gagal (FAILURE)
- Statistik per worker
