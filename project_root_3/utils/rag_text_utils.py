from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple
from logs import get_logger
import re
import html

logger = get_logger("rag_text_utils")

COMMON_ENCODINGS = ["utf-8", "cp1251", "windows-1251", "latin-1", "iso-8859-1"]
MAX_FILE_SIZE_MB = 500

def _smart_read_text(path: Path) -> str:
    """
    Пробует прочитать текстовый файл с помощью популярных кодировок.
    Возвращает содержимое файла или выбрасывает UnicodeDecodeError, если не удалось.
    """
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.error(f"Файл слишком большой (> {MAX_FILE_SIZE_MB} МБ): {path}")
        raise IOError(f"File too large: {path}")
    for encoding in COMMON_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except Exception as e:
            logger.debug(f"Проблема с кодировкой {encoding} для {path}: {e}")
    logger.error(f"Не удалось прочитать файл {path} в поддерживаемых кодировках: {COMMON_ENCODINGS}")
    raise UnicodeDecodeError("all", b'', 0, 1, f"Failed to read {path} with encodings: {COMMON_ENCODINGS}")

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 0,
    max_chunks: Optional[int] = None
) -> List[str]:
    """
    Делит строку на чанки по словам с заданным размером и overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным")
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():  # отбрасываем пустые чанки
            chunks.append(chunk)
        if max_chunks is not None and len(chunks) >= max_chunks:
            logger.info(f"Обрезано по max_chunks={max_chunks}")
            break
    return chunks

def process_text_file_for_rag(
    file_path: Path,
    chunk_size: int = 1000,
    overlap: int = 0,
    max_chunks: Optional[int] = None,
    raise_on_error: bool = False
) -> List[str]:
    """
    Читает текстовый файл и делит его на чанки для RAG.
    """
    try:
        text = _smart_read_text(file_path)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
        logger.info(f"Text file processed for RAG: {file_path.name}, chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"process_text_file_for_rag error: {e}")
        if raise_on_error:
            raise
        return []

def process_text_for_rag(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 0,
    max_chunks: Optional[int] = None
) -> List[str]:
    """
    Делит произвольную строку на чанки для RAG.
    """
    try:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap, max_chunks=max_chunks)
        logger.info(f"Arbitrary text processed for RAG, chunks: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"process_text_for_rag error: {e}")
        return []

def safe_eval(expr: str, variables: Optional[Dict] = None) -> Union[int, float, str]:
    """
    Безопасно вычисляет простое математическое выражение.
    Внимание: не используйте для недоверенного кода!

    :param expr: Выражение для вычисления (строка)
    :param variables: Дополнительные переменные (dict)
    :return: результат вычисления или строка с ошибкой
    """
    import math
    try:
        allowed_names = {"__builtins__": None}
        # Добавим math-функции
        allowed_names.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
        if variables:
            allowed_names.update(variables)
        result = eval(expr, allowed_names)
        logger.info(f"safe_eval: '{expr}' -> {result}")
        return result
    except Exception as e:
        logger.error(f"safe_eval error: {e} (expression: {expr})")
        return f"[Ошибка safe_eval: {e}]"

def remove_invisible(text: str) -> str:
    # Убираем невидимые символы, кроме \n, \t, стандартных пробелов
    return re.sub(r'[^\S\r\n\t ]|[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)

# --- Markdown and HTML normalization for Telegram ---

TELEGRAM_HTML_TAGS = {
    'b': 'b',
    'strong': 'b',
    'i': 'i',
    'em': 'i',
    'u': 'u',
    'ins': 'u',
    's': 's',
    'strike': 's',
    'del': 's',
    'code': 'code',
    'pre': 'pre',
    'a': 'a',
    'span': 'span',  # только с class="tg-spoiler"
}

def markdown_to_telegram_html(text: str) -> str:
    """
    Конвертирует markdown-разметку в html, оставляя только поддерживаемые Telegram-теги.
    Преобразует: **bold**, *italic*, __underline__, ~~strike~~, `code`, [link](url), ---.
    """
    # Жирный/курсив/подчёркнутый/зачёркнутый/код/ссылки
    # Сначала защитим inline code от подстановки
    text = re.sub(r'```(.*?)```', lambda m: f"<pre>{m.group(1)}</pre>", text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'___(.+?)___', r'<u><i>\1</i></u>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)  # *italic*
    text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)        # _italic_
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    # Ссылки [текст](url)
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
    # Разделитель ---
    text = re.sub(r'^---$', r'\n', text, flags=re.MULTILINE)
    # Удалить не поддерживаемые Markdown списки, оставить plain-text
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE) # numbered lists
    return text

def normalize_html_for_telegram(text: str) -> str:
    """
    Оставляет только поддерживаемые Telegram HTML-теги.
    """
    # Заменить аналоги
    tag_map = [
        ('strong', 'b'), ('em', 'i'),
        ('ins', 'u'), ('strike', 's'), ('del', 's')
    ]
    for src, dst in tag_map:
        text = re.sub(rf'<{src}(\s*?)>', f'<{dst}>', text, flags=re.IGNORECASE)
        text = re.sub(rf'</{src}>', f'</{dst}>', text, flags=re.IGNORECASE)
    # span: разрешён только с class="tg-spoiler"
    def filter_span(m):
        tag = m.group(0)
        if 'class="tg-spoiler"' in tag or "class='tg-spoiler'" in tag:
            return tag
        return ''
    text = re.sub(r'<span(?:\s+[^>]*)?>', filter_span, text, flags=re.IGNORECASE)
    text = re.sub(r'</span>', lambda m: m.group(0) if 'tg-spoiler' in text[max(0, m.start()-30):m.start()] else '', text, flags=re.IGNORECASE)
    # Удаляем все неразрешённые теги
    def remove_unsupported_tags(match):
        tag = match.group(1)
        if tag.lower() == 'a':
            return match.group(0)
        if tag.lower() == 'span' and 'tg-spoiler' in match.group(0):
            return match.group(0)
        if tag.lower() in TELEGRAM_HTML_TAGS:
            return match.group(0)
        return ''
    text = re.sub(r'</?([a-zA-Z0-9]+)(\s+[^>]*)?>', remove_unsupported_tags, text)
    return text

def html_escape_telegram(s: str) -> str:
    # Экранируем только &, <, > вне тегов
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def telegram_postprocess(
    text: str,
    max_len: int = 4096
) -> Tuple[str, Dict]:
    """
    Полный пайплайн:
    1. Markdown → HTML
    2. Оставляем только телеграмные теги
    3. Экранируем спецсимволы вне тегов
    4. Ограничиваем длину
    5. meta_info
    """
    meta_info = {}
    # 1. markdown → html
    html_text = markdown_to_telegram_html(text)
    if html_text != text:
        meta_info['markdown_converted'] = True
    # 2. только поддерживаемые теги
    html_text2 = normalize_html_for_telegram(html_text)
    if html_text2 != html_text:
        meta_info['html_normalized'] = True
    # 3. selective escape: экранируем только вне тегов
    TAG_RE = re.compile(r'</?([a-zA-Z0-9]+)(\s+[^>]*)?>')
    parts = []
    last = 0
    for m in TAG_RE.finditer(html_text2):
        start, end = m.span()
        parts.append(html_escape_telegram(html_text2[last:start]))
        parts.append(html_text2[start:end])
        last = end
    parts.append(html_escape_telegram(html_text2[last:]))
    result = ''.join(parts)
    # 4. Ограничиваем длину
    if len(result) > max_len:
        result = result[:max_len-1] + "…"
        meta_info['truncated_for_tg'] = True
    meta_info['final_length'] = len(result)
    return result, meta_info

def prepare_text_for_telegram(
    text: str,
    html_escape: bool = True,
    remove_emoji: bool = False,
    max_len: int = 4096
) -> Tuple[str, Dict]:
    """
    Универсальный pre/post-процессор для Telegram.
    Возвращает текст и meta_info.
    remove_emoji: по умолчанию False, так как эмодзи нужны для Telegram-постов.
    html_escape: игнорируется, т.к. экранирование делается в telegram_postprocess.
    """
    meta_info = {}
    original = text

    # Приведение к str и utf-8
    if not isinstance(text, str):
        try:
            text = text.decode('utf-8')
            meta_info['encoding_fixed'] = True
        except Exception:
            text = str(text)
            meta_info['encoding_forced'] = True

    # Удаление невидимых символов
    cleaned = remove_invisible(text)
    if cleaned != text:
        meta_info['removed_invisible'] = True
    text = cleaned

    # Удаление emoji (не включаем по умолчанию, чтобы не терять стиль Telegram-поста)
    if remove_emoji:
        try:
            import emoji
            cleaned = emoji.replace_emoji(text, replace='')
            if cleaned != text:
                meta_info['removed_emoji'] = True
            text = cleaned
        except ImportError:
            meta_info['emoji_lib_missing'] = True

    # Новый универсальный пайплайн  
    result, meta_info2 = telegram_postprocess(text, max_len=max_len)
    meta_info.update(meta_info2)

    return result, meta_info
