import requests
import json
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
from utils.exceptions import TelegramError
import logging
import time
import html
import traceback
import functools
import re

def get_logger(name: str, logfile: Optional[Union[str, Path]] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        if logfile:
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(sh)
        logger.setLevel(level)
    return logger

logger = get_logger("rag_telegram")

def escape_html(text: str) -> str:
    """
    Экранирует HTML-спецсимволы для Telegram (HTML-mode).
    """
    return html.escape(text, quote=False)

def filter_llm_text(text: str) -> Tuple[str, Dict]:
    """
    Глубокая фильтрация размышлений LLM и мусора, рекурсивно, с поддержкой вложенных тегов и сложных паттернов.
    Удаляет размышления, Reasoning, AI Thoughts, <think>, <reason>, <thought> и их вложения.
    Дополнительно убирает таблицы (markdown и plain), markdown-ссылки, html-теги, лишние переносы строк.
    Возвращает кортеж: (текст, meta_info).
    """
    meta_info = {}
    if not text:
        return "", meta_info
    filtered = text
    # Рекурсивное удаление размышлений в тегах
    patterns = [
        r'<(think|reason|thought)[^>]*>.*?</\1>',  # <think>...</think>, <reason>...</reason>, <thought>...</thought>
        r'<(think|reason|thought)[^>]*>.*',        # незакрытые
        r'(?i)(размышления|reasoning|ai thoughts?|мои размышления|мышление):.*?(\n|$)',  # строковые
    ]
    changed = False
    for patt in patterns:
        # Рекурсивно удаляем до тех пор, пока что-то меняется
        while True:
            new_filtered = re.sub(patt, '', filtered, flags=re.DOTALL | re.IGNORECASE)
            if new_filtered == filtered:
                break
            filtered = new_filtered
            changed = True
    if changed:
        meta_info['filtered_thoughts'] = True

    # Убираем markdown-таблицы (| ... | ... |) и plain-таблицы (разделённые табами)
    table_pattern_md = r"(?:\|[^\n]*\|(?:\n|$))+"
    table_pattern_plain = r"(?:[^\n\t]*\t[^\n]*\n)+"
    filtered_tables = filtered
    filtered = re.sub(table_pattern_md, '', filtered)
    if filtered != filtered_tables:
        meta_info['removed_markdown_table'] = True
    filtered_tables = filtered
    filtered = re.sub(table_pattern_plain, '', filtered)
    if filtered != filtered_tables:
        meta_info['removed_plain_table'] = True

    # Удалить markdown-ссылки [text](url)
    filtered_before = filtered
    filtered = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', filtered)
    if filtered != filtered_before:
        meta_info['removed_md_links'] = True

    # Удалить служебные комментарии типа [RAG ...] или [AI ...]
    filtered_before = filtered
    filtered = re.sub(r'\[(?:RAG|AI)[^]]*\]', '', filtered)
    if filtered != filtered_before:
        meta_info['removed_llm_service_comments'] = True

    # Удалить HTML-теги
    filtered_before = filtered
    filtered = re.sub(r'<[^>]+>', '', filtered)
    if filtered != filtered_before:
        meta_info['removed_html_tags'] = True

    # Удаление лишних переводов строк
    filtered_before = filtered
    filtered = re.sub(r'\n{3,}', '\n\n', filtered)
    if filtered != filtered_before:
        meta_info['cleaned_newlines'] = True

    # Удалить ведущие и завершающие пробелы
    filtered = filtered.strip()

    if filtered != text:
        meta_info['filtered_llm_text'] = True

    return filtered, meta_info

def split_text_for_telegram(text: str, max_len: int = 4096) -> List[str]:
    """
    Делит длинный текст на части максимально допустимой длины для Telegram, учитывая экранирование.
    """
    # Telegram считает длину экранированного текста, поэтому сразу экранируем
    if len(text) <= max_len:
        return [text]
    # Аккуратно бьем по абзацам, если возможно
    parts, current = [], ''
    for paragraph in text.split('\n\n'):
        # учтем +2 на разделитель
        candidate = (('\n\n' if current else '') + paragraph)
        if len(current) + len(candidate) <= max_len:
            current += candidate
        else:
            if current:
                parts.append(current)
            # Если абзац слишком длинный — режем его жёстко
            if len(paragraph) > max_len:
                for i in range(0, len(paragraph), max_len):
                    parts.append(paragraph[i:i + max_len])
                current = ''
            else:
                current = paragraph
    if current:
        parts.append(current)
    return parts

def retry_on_failure(max_retries=3, retry_delay=3.0):
    """
    Декоратор для автоматического повтора при ошибках сетевого уровня.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    tb = traceback.format_exc()
                    logger.warning(f"Telegram API request failed (attempt {attempt}): {e}\n{tb}")
                    time.sleep(retry_delay)
            logger.error(f"Telegram API request failed after {max_retries} attempts: {last_exc}")
            raise TelegramError(f"Telegram API request failed after {max_retries} attempts: {last_exc}") from last_exc
        return wrapper
    return decorator

class TelegramPublisher:
    """
    Публикация сообщений и файлов в Telegram-канал через Bot API.
    Поддерживает отправку текста, изображений, документов, видео, аудио, предпросмотр ссылок, отложенную публикацию.
    """

    def __init__(
        self,
        bot_token: str,
        channel_id: Union[str, int],
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        retry_delay: float = 3.0,
        enable_preview: bool = True
    ):
        """
        :param bot_token: Токен Telegram-бота
        :param channel_id: ID или username канала (например, @my_channel)
        :param logger: Логгер
        :param max_retries: Количество попыток при ошибках сети/Telegram
        :param retry_delay: Задержка между попытками (сек)
        :param enable_preview: Включить предпросмотр ссылок в постах
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_preview = enable_preview
        self.logger = logger or get_logger("rag_telegram")

    @retry_on_failure()
    def _post(self, method: str, data: dict, files: dict = None) -> dict:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        resp = requests.post(url, data=data, files=files, timeout=20)
        try:
            resp.raise_for_status()
        except Exception:
            # Логируем причину ошибки
            self.logger.error(f"Telegram API HTTP error: {resp.text}")
            raise
        result = resp.json()
        if not result.get("ok"):
            self.logger.error(f"Telegram API error: {result}")
            raise TelegramError(f"Telegram API error: {result}")
        return result

    def send_text(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[List[int]]:
        """
        Отправка текстового сообщения в канал с учётом лимита длины.
        Глубокая фильтрация размышлений и таблиц, markdown, html-тегов, аккуратно ведёт лог, поддерживает meta_info.
        :return: список message_id отправленных сообщений или None при ошибке
        """
        filtered_text, meta_info = filter_llm_text(text)
        # Экранирование HTML только после всей фильтрации
        if html_escape and parse_mode == "HTML":
            filtered_text = escape_html(filtered_text)
            meta_info['html_escaped'] = True
        if not filtered_text:
            filtered_text = "Извините, произошла ошибка генерации ответа."
            meta_info['empty_text_substituted'] = True
        parts = split_text_for_telegram(filtered_text)
        message_ids = []
        for idx, part in enumerate(parts):
            data = {
                "chat_id": self.channel_id,
                "text": part,
                "parse_mode": parse_mode,
                "disable_web_page_preview": not (disable_preview if disable_preview is not None else self.enable_preview),
                "disable_notification": silent,
            }
            if reply_to_message_id and idx == 0:
                data["reply_to_message_id"] = reply_to_message_id
            try:
                resp = self._post("sendMessage", data)
                msg_id = resp.get("result", {}).get("message_id")
                log_msg = f"Message part {idx + 1}/{len(parts)} posted to Telegram (id={msg_id})"
                if meta_info:
                    log_msg += f" | meta_info: {meta_info}"
                self.logger.info(log_msg)
                message_ids.append(msg_id)
            except Exception as e:
                self.logger.error(f"Failed to send text message part {idx + 1}: {e}\n{traceback.format_exc()}")
                continue
        return message_ids if message_ids else None

    def send_text_with_button(
        self,
        text: str,
        button_url: str = "https://t.me/Pigment_opercat",
        button_text: str = "Обратная связь",
        parse_mode: str = "HTML",
        disable_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[List[int]]:
        """
        Отправка текстового сообщения с inline-кнопкой.
        """
        filtered_text, meta_info = filter_llm_text(text)
        # Экранирование HTML только после всей фильтрации
        if html_escape and parse_mode == "HTML":
            filtered_text = escape_html(filtered_text)
            meta_info['html_escaped'] = True
        if not filtered_text:
            filtered_text = "Извините, произошла ошибка генерации ответа."
            meta_info['empty_text_substituted'] = True
        parts = split_text_for_telegram(filtered_text)
        message_ids = []
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": button_text, "url": button_url}
                ]
            ]
        }
        for idx, part in enumerate(parts):
            data = {
                "chat_id": self.channel_id,
                "text": part,
                "parse_mode": parse_mode,
                "reply_markup": json.dumps(reply_markup, ensure_ascii=False),
                "disable_web_page_preview": not (disable_preview if disable_preview is not None else self.enable_preview),
                "disable_notification": silent,
            }
            if reply_to_message_id and idx == 0:
                data["reply_to_message_id"] = reply_to_message_id
            try:
                resp = self._post("sendMessage", data)
                msg_id = resp.get("result", {}).get("message_id")
                log_msg = f"Message part {idx + 1}/{len(parts)} with button posted to Telegram (id={msg_id})"
                if meta_info:
                    log_msg += f" | meta_info: {meta_info}"
                self.logger.info(log_msg)
                message_ids.append(msg_id)
            except Exception as e:
                self.logger.error(f"Failed to send text message with button part {idx + 1}: {e}\n{traceback.format_exc()}")
                continue
        return message_ids if message_ids else None

    def send_photo(
        self,
        photo: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        disable_preview: Optional[bool] = None,
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка фото с подписью.
        :param photo: путь к файлу или URL
        :param html_escape: экранировать HTML в подписи
        :return: message_id или None
        """
        if caption:
            caption, _ = filter_llm_text(caption)
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        file_handle = None
        try:
            if isinstance(photo, (str, Path)) and Path(photo).exists():
                file_handle = open(photo, "rb")
                files["photo"] = file_handle
            else:
                data["photo"] = str(photo)
            resp = self._post("sendPhoto", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Photo posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send photo: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if file_handle:
                file_handle.close()

    def send_video(
        self,
        video: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка видеофайла.
        """
        if caption:
            caption, _ = filter_llm_text(caption)
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        file_handle = None
        try:
            if isinstance(video, (str, Path)) and Path(video).exists():
                file_handle = open(video, "rb")
                files["video"] = file_handle
            else:
                data["video"] = str(video)
            resp = self._post("sendVideo", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Video posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send video: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if file_handle:
                file_handle.close()

    def send_audio(
        self,
        audio: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка аудиофайла.
        """
        if caption:
            caption, _ = filter_llm_text(caption)
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        file_handle = None
        try:
            if isinstance(audio, (str, Path)) and Path(audio).exists():
                file_handle = open(audio, "rb")
                files["audio"] = file_handle
            else:
                data["audio"] = str(audio)
            resp = self._post("sendAudio", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Audio posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send audio: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if file_handle:
                file_handle.close()

    def send_document(
        self,
        document: Union[str, Path],
        caption: Optional[str] = None,
        parse_mode: str = "HTML",
        reply_to_message_id: Optional[int] = None,
        silent: bool = False,
        html_escape: bool = True
    ) -> Optional[int]:
        """
        Отправка файла-документа.
        :param document: путь к файлу или URL
        :param html_escape: экранировать HTML в подписи
        :return: message_id или None
        """
        if caption:
            caption, _ = filter_llm_text(caption)
        data = {
            "chat_id": self.channel_id,
            "parse_mode": parse_mode,
            "disable_notification": silent,
        }
        if caption:
            data["caption"] = escape_html(caption) if html_escape and parse_mode == "HTML" else caption
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        files = {}
        file_handle = None
        try:
            if isinstance(document, (str, Path)) and Path(document).exists():
                file_handle = open(document, "rb")
                files["document"] = file_handle
            else:
                data["document"] = str(document)
            resp = self._post("sendDocument", data, files)
            msg_id = resp.get("result", {}).get("message_id")
            self.logger.info(f"Document posted to Telegram (id={msg_id})")
            return msg_id
        except Exception as e:
            self.logger.error(f"Failed to send document: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if file_handle:
                file_handle.close()

    def send_media_group(
        self,
        media: List[dict]
    ) -> Optional[List[int]]:
        """
        Отправка набора медиа (фото/видео) в одном сообщении.
        :param media: список dict с типом ('photo'/'video'), media (file_id/url), caption (optional)
        :return: список message_id или None
        """
        # Фильтруем подписи в медиа
        for item in media:
            if "caption" in item:
                item["caption"], _ = filter_llm_text(item["caption"])
        data = {
            "chat_id": self.channel_id,
            "media": json.dumps(media, ensure_ascii=False),
        }
        try:
            resp = self._post("sendMediaGroup", data)
            results = resp.get("result", [])
            msg_ids = [msg.get("message_id") for msg in results if "message_id" in msg]
            self.logger.info(f"Media group posted to Telegram (messages={msg_ids})")
            return msg_ids
        except Exception as e:
            self.logger.error(f"Failed to send media group: {e}\n{traceback.format_exc()}")
            return None

    @retry_on_failure()
    def check_connection(self) -> bool:
        """
        Проверка связи с Telegram Bot API (getMe).
        """
        url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            self.logger.info("Telegram bot connection OK")
            return True
        else:
            self.logger.error("Telegram bot connection failed")
            return False

    def delayed_post(
        self,
        text: str,
        delay_sec: float,
        **kwargs
    ) -> Optional[List[int]]:
        """
        Отправка сообщения с задержкой.
        """
        self.logger.info(f"Delaying message post for {delay_sec} seconds...")
        time.sleep(delay_sec)
        return self.send_text(text, **kwargs)
