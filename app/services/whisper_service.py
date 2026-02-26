"""
WhisperService — Faster Whisper transcription service.

WINDOWS DLL CONFLICT FIX:
Singleton dihapus. Model di-load per request agar tidak ada konflik
dengan Docling yang juga menggunakan PyTorch.

Jika WHISPER_KEEP_IN_MEMORY=True di .env, model tetap di-cache di memori
(lebih cepat) tapi JANGAN jalankan Docling setelahnya dalam proses yang sama.

Default: WHISPER_KEEP_IN_MEMORY=False (aman untuk Windows, sedikit lebih lambat)
"""
import os
import gc
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, List
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Cache model di memori hanya jika setting mengizinkan
_cached_model = None
_cached_model_size = None


def _get_model():
    """
    Load atau ambil model dari cache.
    Cache digunakan hanya jika WHISPER_KEEP_IN_MEMORY=True.
    """
    global _cached_model, _cached_model_size

    model_size = settings.WHISPER_MODEL_SIZE
    keep_in_memory = getattr(settings, "WHISPER_KEEP_IN_MEMORY", False)

    if keep_in_memory and _cached_model is not None and _cached_model_size == model_size:
        logger.debug("[Whisper] Using cached model")
        return _cached_model, False  # model, should_release

    logger.info(
        f"[Whisper] Loading model: {model_size} "
        f"[{settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE}]"
    )

    from faster_whisper import WhisperModel
    model = WhisperModel(
        model_size,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
    )
    logger.info("✅ Whisper model loaded")

    if keep_in_memory:
        _cached_model = model
        _cached_model_size = model_size
        return model, False  # di-cache, tidak perlu release

    return model, True  # tidak di-cache, harus di-release setelah selesai


def _release_model(model, should_release: bool):
    """Release model dari memori jika WHISPER_KEEP_IN_MEMORY=False."""
    if not should_release:
        return
    try:
        del model
        gc.collect()
        logger.debug("[Whisper] Model released from memory")
    except Exception:
        pass


class WhisperService:
    """
    WhisperService tanpa singleton.
    Setiap instance load model fresh → tidak ada konflik DLL dengan Docling.
    """

    def __init__(self):
        self.denoiser = None
        if settings.DENOISE_ENABLED:
            self._init_denoiser()

    def _init_denoiser(self):
        try:
            import noisereduce as nr
            self.denoiser = nr
            logger.info("✅ Denoiser (noisereduce) loaded.")
        except ImportError:
            logger.warning("⚠️  noisereduce not installed. Denoising disabled.")

    # Backward compat — endpoint masih pakai .get_instance()
    @classmethod
    def get_instance(cls) -> "WhisperService":
        return cls()

    def _denoise_audio(self, audio_path: str) -> str:
        if self.denoiser is None:
            return audio_path

        try:
            import soundfile as sf
            import librosa

            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)

            if audio_data.ndim == 2:
                logger.debug("Denoising stereo audio (per channel)")
                denoised_channels = []
                for ch in range(audio_data.shape[0]):
                    reduced = self.denoiser.reduce_noise(
                        y=audio_data[ch],
                        sr=sample_rate,
                        prop_decrease=settings.DENOISE_PROP_DECREASE,
                        stationary=settings.DENOISE_STATIONARY,
                    )
                    denoised_channels.append(reduced)
                denoised = np.array(denoised_channels)
            else:
                logger.debug("Denoising mono audio")
                denoised = self.denoiser.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    prop_decrease=settings.DENOISE_PROP_DECREASE,
                    stationary=settings.DENOISE_STATIONARY,
                )

            stem = Path(audio_path).stem
            denoised_path = os.path.join(
                os.path.dirname(audio_path), f"{stem}_denoised.wav"
            )

            if denoised.ndim == 2:
                sf.write(denoised_path, denoised.T, sample_rate)
            else:
                sf.write(denoised_path, denoised, sample_rate)

            logger.debug(f"Audio denoised → {denoised_path} (sr={sample_rate})")
            return denoised_path

        except Exception as e:
            logger.warning(f"Denoising failed: {e}. Using original audio.")
            return audio_path

    def transcribe_to_dict(
        self,
        audio_path: str,
        language: str = "auto",
        beam_size: int = None,
        condition_on_previous_text: bool = None,
        chunk_length_s: int = None,
        no_speech_threshold: float = None,
    ) -> dict:
        """
        Transcribe audio file → dict dengan segments & metadata.

        Model di-load di sini (bukan __init__) agar tidak ada
        DLL yang ter-lock saat Docling atau service lain di-load.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        _beam_size = beam_size if beam_size is not None else settings.WHISPER_BEAM_SIZE
        _cond_prev = condition_on_previous_text if condition_on_previous_text is not None else settings.WHISPER_CONDITION_ON_PREV
        _chunk_len = chunk_length_s if chunk_length_s is not None else settings.WHISPER_CHUNK_LENGTH_S
        _no_speech = no_speech_threshold if no_speech_threshold is not None else settings.WHISPER_NO_SPEECH_THRESHOLD

        start_time = time.time()
        logger.info(
            f"[Whisper] Transcribing: {Path(audio_path).name} "
            f"[lang={language}, beam={_beam_size}]"
        )

        process_path = self._denoise_audio(audio_path)
        denoised = process_path != audio_path

        model = None
        should_release = False

        try:
            model, should_release = _get_model()

            detect_lang = None if language == "auto" else language
            segments_gen, info = model.transcribe(
                process_path,
                language=detect_lang,
                beam_size=_beam_size,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                },
                condition_on_previous_text=_cond_prev,
                no_speech_threshold=_no_speech,
                compression_ratio_threshold=2.4,
                chunk_length=_chunk_len,
                word_timestamps=False,
            )

            all_segments: List[dict] = []
            full_text: List[str] = []
            seg_count = 0
            last_log = start_time

            for seg in segments_gen:
                all_segments.append({
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                })
                full_text.append(seg.text.strip())
                seg_count += 1

                now = time.time()
                if now - last_log >= 5:
                    elapsed = now - start_time
                    if info.duration and seg.end:
                        pct = min(seg.end / info.duration * 100, 100)
                        speed = seg.end / elapsed if elapsed > 0 else 0
                        logger.debug(
                            f"[Whisper] Progress: {pct:.1f}% | segs={seg_count} | speed={speed:.1f}x"
                        )
                    last_log = now

            transcription_time = round(time.time() - start_time, 2)
            realtime_factor = round(info.duration / transcription_time, 2) if transcription_time > 0 else 0

            logger.info(
                f"✅ Transcription done: {seg_count} segs | "
                f"lang={info.language} ({info.language_probability:.2f}) | "
                f"duration={info.duration:.1f}s | elapsed={transcription_time}s | "
                f"RTF={realtime_factor}x"
            )

            return {
                "source_file": Path(audio_path).name,
                "content": " ".join(full_text),
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "transcription_time": transcription_time,
                "realtime_factor": realtime_factor,
                "segments": all_segments,
            }

        finally:
            if model is not None:
                _release_model(model, should_release)
            if denoised and os.path.exists(process_path):
                os.remove(process_path)