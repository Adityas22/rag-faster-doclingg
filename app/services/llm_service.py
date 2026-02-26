"""
LLMService — Text generation via Llama (Ollama) atau Gemini (fallback).
"""
import json
import asyncio
import re
from typing import Optional
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(self):
        self.use_ollama = self._check_ollama()

    def _check_ollama(self) -> bool:
        try:
            import requests
            r = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"✅ Ollama available. Models: {models}")
                return True
        except Exception:
            pass
        logger.info("⚠️  Ollama unavailable. Falling back to Gemini.")
        return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate teks, coba Ollama dulu, fallback ke Gemini."""
        if self.use_ollama:
            try:
                return await self._generate_ollama(prompt, system_prompt, max_tokens, temperature)
            except Exception as e:
                logger.warning(f"Ollama error, falling back to Gemini: {e}")

        return await self._generate_gemini(prompt, system_prompt, max_tokens, temperature)

    async def _generate_ollama(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
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
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                data = await resp.json()
                return data.get("response", "")

    async def _generate_gemini(
        self, prompt: str, system_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name=settings.GEMINI_GENERATE_MODEL,
            system_instruction=system_prompt or None,
        )

        full_prompt = prompt
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            ),
        )
        return response.text

    def parse_json_response(self, text: str) -> dict | list:
        """
        Parsing response LLM yang mungkin mengandung markdown code block.
        Contoh: ```json\n{...}\n```
        """
        # Remove markdown code block
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
        text = text.strip("`").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}\nRaw text: {text[:200]}")
            raise ValueError(f"LLM response bukan valid JSON: {e}")
