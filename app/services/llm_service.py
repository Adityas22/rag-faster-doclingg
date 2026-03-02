"""
LLMService — Llama (Ollama) PRIMARY, Gemini fallback.
"""
import json
import asyncio
import re
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
OLLAMA_TIMEOUT = int(getattr(settings, "OLLAMA_TIMEOUT", 120))


class LLMService:

    def __init__(self):
        self.ollama_available = self._check_ollama()
        if self.ollama_available:
            logger.info(f"✅ LLM Primary: Ollama ({settings.OLLAMA_MODEL})")
        else:
            logger.info("⚠️  Ollama tidak tersedia → fallback Gemini")

    def _check_ollama(self) -> bool:
        try:
            import requests
            r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"✅ Ollama models: {models}")
                model_available = any(
                    settings.OLLAMA_MODEL in m or m.startswith(settings.OLLAMA_MODEL.split(":")[0])
                    for m in models
                )
                if not model_available:
                    self._pull_model(settings.OLLAMA_MODEL)
                return True
        except Exception as e:
            logger.warning(f"⚠️  Ollama check failed: {e}")
        return False

    def _pull_model(self, model_name: str):
        try:
            import requests
            logger.info(f"[Ollama] Pulling {model_name}...")
            r = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300,
            )
            if r.status_code == 200:
                logger.info(f"[Ollama] ✅ {model_name} siap")
        except Exception as e:
            logger.warning(f"[Ollama] Pull error: {e}")

    async def generate(self, prompt: str, system_prompt: str = "",
                       max_tokens: int = 2048, temperature: float = 0.7) -> str:
        if self.ollama_available:
            try:
                return await self._generate_ollama(prompt, system_prompt, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"[LLM] Ollama error → fallback Gemini: {e}")

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("Ollama tidak tersedia dan GEMINI_API_KEY belum diset.")
        return await self._generate_gemini(prompt, system_prompt, max_tokens, temperature)

    async def _generate_ollama(self, prompt, system_prompt, max_tokens, temperature) -> str:
        import aiohttp
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=OLLAMA_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama HTTP {resp.status}: {text[:200]}")
                data = await resp.json()
                return data.get("response", "").strip()

    async def _generate_gemini(self, prompt, system_prompt, max_tokens, temperature) -> str:
        import google.generativeai as genai
        logger.info("[LLM] Using Gemini fallback")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_GENERATE_MODEL,
            system_instruction=system_prompt or None,
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            ),
        )
        return response.text

    def parse_json_response(self, text: str) -> dict | list:
        text = re.sub(r"```(?:json)?\n?", "", text).strip().strip("`").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response bukan valid JSON: {e}")