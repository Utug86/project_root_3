from pathlib import Path
from typing import List, Union, Optional, Dict
from logs import get_logger

logger = get_logger("rag_text_utils")

COMMON_ENCODINGS = ["utf-8", "cp1251", "windows-1251", "latin-1", "iso-8859-1"]
MAX_FILE_SIZE_MB = 50

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
