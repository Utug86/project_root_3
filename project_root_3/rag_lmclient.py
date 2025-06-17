import re
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import logging

from rag_langchain_tools import enrich_context_with_tools
from rag_prompt_utils import get_prompt_parts

class LMClientException(Exception):
    """Базовое исключение LLM клиента."""
    pass

class LMClient:
    """
    Асинхронный клиент для генерации текстов через LLM API.
    Гарантирует асинхронность, контроль длины, SRP, устойчивость к ошибкам.
    """

    def __init__(
        self,
        retriever: Any,
        data_dir: Union[str, Path],
        inform_dir: Union[str, Path],
        logger: logging.Logger,
        model_url: str,
        model_name: str,
        max_tokens: int = 1024,
        max_chars: int = 2600,
        max_attempts: int = 3,
        temperature: float = 0.7,
        timeout: int = 40,
        history_lim: int = 3,
        system_msg: Optional[str] = None
    ):
        self.retriever = retriever
        self.data_dir = Path(data_dir)
        self.inform_dir = Path(inform_dir)
        self.logger = logger

        self.model_url = model_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.max_attempts = max_attempts
        self.temperature = temperature
        self.timeout = timeout
        self.history_lim = history_lim
        self.system_msg = system_msg or "Вы — эксперт по бровям и ресницам."

    async def generate(
        self, 
        topic: str, 
        uploadfile: Optional[str] = None,
        max_chars: Optional[int] = None
    ) -> str:
        """
        Генерирует текст по теме с использованием контекста и инструментов.
        :param topic: Тема для генерации
        :param uploadfile: путь к прикреплённому файлу (если нужен в prompt)
        :param max_chars: ограничение на количество символов (опционально, перекрывает self.max_chars)
        :return: Ответ LLM (или строка с ошибкой)
        """
        prev_max_chars = self.max_chars
        if max_chars is not None:
            if not isinstance(max_chars, int) or max_chars <= 0:
                self.logger.error(f"Некорректное значение max_chars: {max_chars}")
                return "[Ошибка: некорректный лимит символов]"
            self.max_chars = max_chars
        try:
            context = await self._get_full_context(topic)
            prompt = self._build_prompt(topic, context, uploadfile)
            return await self._request_llm_with_retries(topic, prompt)
        except Exception as e:
            self.logger.error(f"Critical error in generate: {e}")
            return "[Критическая ошибка генерации]"
        finally:
            # Всегда восстанавливаем лимит
            self.max_chars = prev_max_chars

    async def _get_full_context(self, topic: str) -> str:
        """
        Получает и обогащает контекст по теме.
        """
        try:
            if asyncio.iscoroutinefunction(getattr(self.retriever, "retrieve", None)):
                ctx = await self.retriever.retrieve(topic)
            else:
                ctx = self.retriever.retrieve(topic)
            ctx = enrich_context_with_tools(topic, ctx, self.inform_dir, enforce=True)
            return ctx
        except Exception as e:
            self.logger.warning(f"Ошибка получения/обогащения контекста: {e}")
            return ""

    def _build_prompt(
        self, 
        topic: str, 
        context: str, 
        uploadfile: Optional[str] = None
    ) -> str:
        """
        Генерирует промпт для LLM.
        """
        try:
            return get_prompt_parts(self.data_dir, topic, context, uploadfile=uploadfile)
        except Exception as e:
            self.logger.error(f"Ошибка генерации промпта: {e}")
            prompt_file = self.data_dir / 'prompt.txt'
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                return prompt_template.replace('{TOPIC}', topic).replace('{CONTEXT}', context)
            else:
                return f"{topic}\n\n{context}"

    async def _request_llm_with_retries(self, topic: str, prompt: str) -> str:
        """
        Выполняет несколько попыток генерации текста через LLM API с контролем длины и истории.
        """
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": prompt}
        ]
        for attempt in range(self.max_attempts):
            try:
                text = await self._request_llm(messages)
                text = self._postprocess(text)
                if len(text) <= self.max_chars:
                    self.logger.info(f"Generated text length: {len(text)} chars")
                    return text
                # Если слишком длинно — добавить в историю и просить сжать
                if attempt < self.max_attempts - 1:
                    messages = self._update_history(messages, text)
                else:
                    self.logger.warning(f"Force truncating text from {len(text)} to {self.max_chars} chars")
                    if self.max_chars > 10:
                        return text[:self.max_chars-10] + "..."
                    else:
                        return text[:self.max_chars]
            except asyncio.TimeoutError as e:
                self.logger.error(f"Timeout in attempt {attempt+1}: {e}")
            except aiohttp.ClientResponseError as e:
                # Обработка возможного rate limit (HTTP 429)
                if getattr(e, "status", None) == 429:
                    self.logger.warning(f"LLM API rate limited (HTTP 429) on attempt {attempt+1}")
                    await asyncio.sleep(5)
                else:
                    self.logger.error(f"LLM request ClientResponseError in attempt {attempt+1}: {e}")
            except Exception as e:
                self.logger.error(f"LLM request error in attempt {attempt+1}: {e}")
            await asyncio.sleep(2)
        return "[Ошибка: превышено количество попыток генерации]"

    async def _request_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Асинхронно отправляет запрос к LLM API и получает результат.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.model_url, json=payload) as resp:
                if resp.status == 429:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=resp.status, message="Rate limit",
                        headers=resp.headers
                    )
                if resp.status != 200:
                    raise LMClientException(f"LLM API error: HTTP {resp.status}")
                data = await resp.json()
                if 'choices' not in data or not data['choices']:
                    raise LMClientException("Invalid LLM response format")
                return data['choices'][0]['message']['content'].strip()

    def _postprocess(self, text: str) -> str:
        """
        Глубокая постобработка LLM-ответа: удаляет markdown-заголовки, разделители, markdown-таблицы, ссылки, html-теги, служебные LLM-комменты, множественные переводы строк и лишние пробелы.
        """
        if not text:
            return ""
        # Удалить markdown-заголовки (##, ###, ####)
        text = re.sub(r"(?m)^#{2,}.*$", "", text)
        # Удалить markdown-разделители --- или ***
        text = re.sub(r"(?m)^(-{3,}|\*{3,})$", "", text)
        # Удалить markdown-таблицы (| ... |)
        text = re.sub(r"(?:\|[^\n]*\|(?:\n|$))+", "", text)
        # Удалить plain таблицы (разделённые табами)
        text = re.sub(r"(?:[^\n\t]*\t[^\n]*\n)+", "", text)
        # Удалить markdown-ссылки [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        # Удалить служебные комментарии типа [RAG ...] или [AI ...]
        text = re.sub(r'\[(?:RAG|AI)[^]]*\]', '', text)
        # Удалить HTML-теги
        text = re.sub(r'<[^>]+>', '', text)
        # Удалить размышления и reasoning-фрагменты
        text = re.sub(r'(?i)(размышления|reasoning|ai thoughts?|мои размышления|мышление):.*?(\n|$)', '', text)
        text = re.sub(r'<(think|reason|thought)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<(think|reason|thought)[^>]*>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Удалить фразы AI (as an ai language model...)
        text = re.sub(
            r"(as an ai language model|i am an ai language model|я искусственный интеллект|как искусственный интеллект)[\.,]?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        # Удалить множественные переводы строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Ведущие и завершающие пробелы
        text = text.strip()
        return text

    def _update_history(self, messages: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
        """
        Обновляет историю сообщений для запроса к LLM (ограничивает по self.history_lim).
        """
        if self.history_lim <= 0:
            self.logger.warning(f"history_lim <= 0: диалоговая история не будет сохраняться.")
            return messages
        # Добавляем assistant/user
        history = messages[:]
        history.append({"role": "assistant", "content": text})
        history.append({
            "role": "user",
            "content": f"Текст слишком длинный ({len(text)}>{self.max_chars}), сократи до {self.max_chars} символов."
        })
        sysm, rest = history[0], history[1:]
        # Берём последние пары user/assistant (history_lim*2), не нарушая структуру
        last_msgs = []
        for m in reversed(rest):
            if len(last_msgs) >= self.history_lim * 2:
                break
            last_msgs.insert(0, m)
        return [sysm] + last_msgs
