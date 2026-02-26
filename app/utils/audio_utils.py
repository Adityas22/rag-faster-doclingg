"""
Audio Utilities — Preprocessing, validation, format conversion.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """
    Ambil informasi audio file menggunakan pydub.

    Returns:
        {duration_seconds, sample_rate, channels, format, file_size_mb}
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File tidak ditemukan: {audio_path}")

    ext = Path(audio_path).suffix.lower()
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)

    info = {
        "file_name": Path(audio_path).name,
        "format": ext.lstrip("."),
        "file_size_mb": round(file_size_mb, 2),
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
    }

    try:
        from pydub import AudioSegment
        fmt = ext.lstrip(".")
        if fmt == "mp3":
            audio = AudioSegment.from_mp3(audio_path)
        elif fmt == "wav":
            audio = AudioSegment.from_wav(audio_path)
        elif fmt == "ogg":
            audio = AudioSegment.from_ogg(audio_path)
        elif fmt in ("m4a", "aac"):
            audio = AudioSegment.from_file(audio_path, format=fmt)
        else:
            audio = AudioSegment.from_file(audio_path)

        info["duration_seconds"] = round(len(audio) / 1000, 2)
        info["sample_rate"] = audio.frame_rate
        info["channels"] = audio.channels
        info["duration_formatted"] = _format_duration(info["duration_seconds"])

    except Exception as e:
        logger.warning(f"pydub info failed for {audio_path}: {e}")

    return info


def convert_to_wav(audio_path: str, output_dir: str = None) -> str:
    """
    Konversi audio ke WAV 16kHz mono (optimal untuk Whisper).
    Membutuhkan ffmpeg terinstall.

    Returns:
        Path ke file WAV hasil konversi
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File tidak ditemukan: {audio_path}")

    output_dir = output_dir or os.path.dirname(audio_path)
    wav_path = os.path.join(
        output_dir, Path(audio_path).stem + "_16k.wav"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ar", "16000",   # 16kHz sample rate
        "-ac", "1",       # mono
        "-f", "wav",
        wav_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        logger.info(f"Converted to WAV: {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install with: apt install ffmpeg")


def validate_audio_file(audio_path: str, max_size_mb: int = 100) -> bool:
    """Validasi file audio: ekstensi + ukuran."""
    ext = Path(audio_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Format tidak didukung: {ext}. Gunakan: {SUPPORTED_FORMATS}")

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File terlalu besar: {size_mb:.1f}MB (maks {max_size_mb}MB)")

    return True


def _format_duration(seconds: float) -> str:
    """Format detik menjadi HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_audio(audio_path: str, segment_duration_ms: int = 300_000) -> list[str]:
    """
    Pecah audio panjang menjadi segmen-segmen (default 5 menit).
    Berguna untuk audio > 30 menit agar tidak OOM.

    Returns:
        List of paths ke file segmen
    """
    from pydub import AudioSegment

    ext = Path(audio_path).suffix.lower().lstrip(".")
    audio = AudioSegment.from_file(audio_path)
    total_ms = len(audio)

    if total_ms <= segment_duration_ms:
        return [audio_path]  # Tidak perlu split

    segments = []
    base_name = Path(audio_path).stem
    out_dir = os.path.dirname(audio_path)

    for i, start in enumerate(range(0, total_ms, segment_duration_ms)):
        end = min(start + segment_duration_ms, total_ms)
        segment = audio[start:end]
        seg_path = os.path.join(out_dir, f"{base_name}_seg{i:03d}.wav")
        segment.export(seg_path, format="wav")
        segments.append(seg_path)
        logger.debug(f"Segment {i}: {start/1000:.0f}s - {end/1000:.0f}s → {seg_path}")

    logger.info(f"Audio split into {len(segments)} segments")
    return segments
