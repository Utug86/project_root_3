from pathlib import Path
import logging
from typing import Optional, Any
import pandas as pd
from bs4 import BeautifulSoup

from logs import get_logger

logger = get_logger("rag_file_utils")

# --- Dynamic imports ---
def _try_import_docx() -> Optional[Any]:
    try:
        import docx
        return docx
    except ImportError:
        logger.warning("python-docx не установлен. Форматы .docx будут проигнорированы.")
        return None

def _try_import_pypdf() -> Optional[Any]:
    try:
        import pypdf
        return pypdf
    except ImportError:
        logger.warning("pypdf не установлен. Форматы .pdf будут проигнорированы.")
        return None

def _try_import_textract() -> Optional[Any]:
    try:
        import textract
        return textract
    except ImportError:
        logger.warning("textract не установлен. Некоторые форматы могут быть не поддержаны.")
        return None

DOCX = _try_import_docx()
PDF = _try_import_pypdf()
TEXTRACT = _try_import_textract()

COMMON_ENCODINGS = ["utf-8", "cp1251", "windows-1251", "latin-1", "iso-8859-1"]
MAX_FILE_SIZE_MB = 100
MAX_EXTRACTED_TEXT_LEN = 5_000_000  # ~5 МБ текста, safeguard

def _smart_read_text(path: Path) -> str:
    """
    Пробует прочитать текстовый файл с помощью популярных кодировок.
    Возвращает содержимое файла или спец. сообщение при ошибке.
    """
    for encoding in COMMON_ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            if len(text) > MAX_EXTRACTED_TEXT_LEN:
                logger.warning(f"Извлеченный текст слишком большой (>5МБ), обрезаем до лимита.")
                return text[:MAX_EXTRACTED_TEXT_LEN]
            return text
        except Exception as e:
            logger.debug(f"Проблема с кодировкой {encoding} для {path}: {e}")
    logger.error(f"Не удалось прочитать файл {path} в поддерживаемых кодировках: {COMMON_ENCODINGS}")
    return "[Ошибка чтения файла: неподдерживаемая кодировка]"

def extract_text_from_file(path: Path) -> str:
    """
    Универсальный парсер для извлечения текста из файлов различных форматов.
    Поддерживает: txt, html, csv, xlsx, xlsm, docx, doc, pdf.
    Возвращает текст или специальную метку при ошибке/неподдерживаемом формате.
    """
    ext = path.suffix.lower()
    if not path.exists():
        logger.error(f"Файл не найден: {path}")
        return "[Файл не найден]"

    filesize = path.stat().st_size
    if filesize > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.warning(f"Файл слишком большой (> {MAX_FILE_SIZE_MB} МБ): {path}")
        return "[Файл слишком большой для обработки]"

    try:
        if ext == ".txt":
            logger.info(f"Extracting text from TXT file: {path}")
            return _smart_read_text(path)

        elif ext == ".html":
            logger.info(f"Extracting text from HTML file: {path}")
            html_content = _smart_read_text(path)
            try:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator=" ")
                if len(text) > MAX_EXTRACTED_TEXT_LEN:
                    logger.warning("Извлеченный текст из HTML слишком большой, обрезаем.")
                    return text[:MAX_EXTRACTED_TEXT_LEN]
                return text
            except Exception as e:
                logger.error(f"Ошибка парсинга HTML: {e}")
                return "[Ошибка парсинга HTML]"

        elif ext == ".csv":
            logger.info(f"Extracting text from CSV file: {path}")
            try:
                df = pd.read_csv(path)
                csv_text = df.to_csv(sep="\t", index=False)
                if len(csv_text) > MAX_EXTRACTED_TEXT_LEN:
                    logger.warning("Извлеченный текст из CSV слишком большой, обрезаем.")
                    return csv_text[:MAX_EXTRACTED_TEXT_LEN]
                return csv_text
            except Exception as e:
                logger.warning(f"Ошибка чтения CSV через pandas: {e}. Пробуем как текст.")
                return _smart_read_text(path)

        elif ext in [".xlsx", ".xls", ".xlsm"]:
            logger.info(f"Extracting text from Excel file: {path}")
            try:
                df = pd.read_excel(path)
                csv_text = df.to_csv(sep="\t", index=False)
                if len(csv_text) > MAX_EXTRACTED_TEXT_LEN:
                    logger.warning("Извлеченный текст из Excel слишком большой, обрезаем.")
                    return csv_text[:MAX_EXTRACTED_TEXT_LEN]
                return csv_text
            except Exception as e:
                logger.error(f"Ошибка чтения Excel через pandas: {e}")
                return "[Ошибка чтения Excel]"

        elif ext == ".docx":
            logger.info(f"Extracting text from DOCX file: {path}")
            if DOCX is not None:
                try:
                    doc = DOCX.Document(path)
                    paragraphs = [p.text for p in doc.paragraphs]
                    text = "\n".join(paragraphs)
                    if len(text) > MAX_EXTRACTED_TEXT_LEN:
                        logger.warning("Извлеченный текст из DOCX слишком большой, обрезаем.")
                        return text[:MAX_EXTRACTED_TEXT_LEN]
                    return text
                except Exception as e:
                    logger.error(f"Ошибка чтения DOCX: {e}")
                    return "[Ошибка чтения DOCX]"
            else:
                logger.warning(f"Модуль python-docx не установлен. DOCX не поддерживается.")
                return "[Формат DOCX не поддерживается]"

        elif ext == ".doc":
            logger.info(f"Extracting text from DOC file: {path}")
            if TEXTRACT is not None:
                try:
                    text = TEXTRACT.process(str(path)).decode("utf-8", errors="ignore")
                    if len(text) > MAX_EXTRACTED_TEXT_LEN:
                        logger.warning("Извлеченный текст из DOC слишком большой, обрезаем.")
                        return text[:MAX_EXTRACTED_TEXT_LEN]
                    return text
                except Exception as e:
                    logger.error(f"Ошибка чтения DOC через textract: {e}")
                    return "[Ошибка чтения DOC]"
            else:
                logger.warning(f"textract не установлен. DOC не поддерживается.")
                return "[Формат DOC не поддерживается]"

        elif ext == ".pdf":
            logger.info(f"Extracting text from PDF file: {path}")
            text = ""
            if PDF is not None:
                try:
                    with open(path, "rb") as f:
                        reader = PDF.PdfReader(f)
                        page_texts = []
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                page_texts.append(page_text)
                        text = "\n".join(page_texts)
                        if len(text) > MAX_EXTRACTED_TEXT_LEN:
                            logger.warning("Извлеченный текст из PDF слишком большой, обрезаем.")
                            text = text[:MAX_EXTRACTED_TEXT_LEN]
                        return text
                except Exception as e:
                    logger.warning(f"Ошибка чтения PDF через pypdf: {e}. Пробуем textract.")
            if TEXTRACT is not None:
                try:
                    text = TEXTRACT.process(str(path)).decode("utf-8", errors="ignore")
                    if len(text) > MAX_EXTRACTED_TEXT_LEN:
                        logger.warning("Извлеченный текст из PDF (textract) слишком большой, обрезаем.")
                        text = text[:MAX_EXTRACTED_TEXT_LEN]
                    return text
                except Exception as e:
                    logger.error(f"Ошибка чтения PDF через textract: {e}")
                    return "[Ошибка чтения PDF]"
            logger.warning(f"pypdf и textract не установлены. PDF не поддерживается.")
            return "[Формат PDF не поддерживается]"

        else:
            logger.warning(f"Неподдерживаемый тип файла: {path}")
            return f"[Неподдерживаемый тип файла: {ext}]"

    except Exception as e:
        logger.error(f"Критическая ошибка при извлечении текста из {path}: {e}")
        return "[Критическая ошибка при извлечении текста]"

def clean_html_from_cell(cell_value: Any) -> str:
    """
    Очищает строку/ячейку от HTML-тегов.
    """
    if isinstance(cell_value, str):
        try:
            return BeautifulSoup(cell_value, "html.parser").get_text(separator=" ")
        except Exception as e:
            logger.warning(f"Ошибка очистки HTML в ячейке: {e}")
            return cell_value
    return str(cell_value)
