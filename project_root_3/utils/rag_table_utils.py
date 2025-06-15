import pandas as pd
from pathlib import Path
from rag_file_utils import clean_html_from_cell
from typing import Any, Dict, List, Optional, Union
import logging
import os

# --- 1. Логгер и конфигурация ---
logger = logging.getLogger("rag_table_utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_table_utils] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- 2. Константы ---
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".xlsm", ".tsv"]
MAX_FILE_SIZE_MB = 50
MAX_OUTPUT_ROWS = 3000
MAX_OUTPUT_COLS = 50  # ограничение на количество столбцов в preview

# --- 3. Вспомогательные функции ---

def _check_file(path: Path) -> None:
    if not isinstance(path, Path):
        raise TypeError("file_path должен быть экземпляром pathlib.Path")
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Указанный путь не является файлом: {path}")
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise IOError(f"Файл слишком большой (> {MAX_FILE_SIZE_MB} МБ): {path}")

def _read_table(file_path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Не поддерживаемый формат файла: {ext}")
    if ext == ".csv":
        return pd.read_csv(file_path, usecols=columns)
    elif ext == ".tsv":
        return pd.read_csv(file_path, sep="\t", usecols=columns)
    elif ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(file_path, usecols=columns)
    else:
        raise ValueError("Неизвестный формат файла (логическая ошибка)")

def _format_row(row: pd.Series, colnames: List[str]) -> str:
    items = []
    for col in colnames:
        val = row[col]
        if pd.isna(val):
            val_str = ""
        else:
            val_str = str(val)
        items.append(f"{col}: {val_str}")
    return " | ".join(items)

# --- 4. Основные функции ---

def process_table_for_rag(
    file_path: Path,
    columns: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    add_headers: bool = True,
    row_delim: str = "\n",
    max_rows: int = MAX_OUTPUT_ROWS
) -> str:
    """
    Читает и форматирует таблицу для подачи в RAG/LLM.
    Поддержка: csv, xlsx, xls, xlsm, tsv.
    :param file_path: путь до файла
    :param columns: список нужных столбцов (или None — все)
    :param filter_expr: pandas query-выражение для фильтрации строк
    :param add_headers: добавить ли строку с заголовками
    :param row_delim: разделитель строк
    :param max_rows: максимальное число строк в выводе
    :return: строка для подачи в LLM/RAG
    """
    try:
        _check_file(file_path)
        logger.info(f"Processing table for RAG: {file_path.name}")
        df = _read_table(file_path, columns=columns)
        logger.info(f"Table read: shape={df.shape}")

        if filter_expr:
            try:
                df = df.query(filter_expr)
                logger.info(f"Filter applied: '{filter_expr}', shape={df.shape}")
            except Exception as e:
                logger.error(f"Ошибка фильтрации (filter_expr): {e}")
                return f"[Ошибка фильтрации таблицы]: {e}"

        if df.empty:
            logger.warning("Пустая таблица после фильтрации/чтения")
            return "[Пустая таблица после фильтрации/чтения]"

        # Ограничение по строкам
        if len(df) > max_rows:
            logger.warning(f"Обрезка таблицы по max_rows={max_rows} (было {len(df)})")
            df = df.iloc[:max_rows]

        # Очищаем HTML в каждой ячейке
        for col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: clean_html_from_cell(x) if pd.notna(x) else x)
            except Exception as e:
                logger.warning(f"Ошибка очистки HTML в столбце {col}: {e}")

        colnames = list(df.columns)
        rows = []
        for idx, row in df.iterrows():
            rows.append(_format_row(row, colnames))

        result_lines = []
        if add_headers:
            result_lines.append(" | ".join(colnames))
        result_lines.extend(rows)

        logger.info(f"Table processed for RAG: {file_path.name}, rows: {len(rows)}")
        return row_delim.join(result_lines)
    except Exception as e:
        logger.error(f"process_table_for_rag error: {e}")
        return f"[Ошибка обработки таблицы для RAG]: {e}"

def analyze_table(
    table_path: Path,
    info_query: Optional[dict] = None,
    max_rows: int = 18,
    max_cols: int = 10
) -> Dict[str, Any]:
    """
    Анализирует табличный файл: возвращает preview, типы, статистику.
    info_query может содержать параметры: columns, summary, filter_expr.
    """
    try:
        _check_file(table_path)
        ext = table_path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(table_path)
        elif ext == ".tsv":
            df = pd.read_csv(table_path, sep="\t")
        elif ext in [".xlsx", ".xls", ".xlsm"]:
            df = pd.read_excel(table_path)
        else:
            return {"error": f"Не поддерживаемый формат файла: {ext}"}
        
        logger.info(f"Таблица прочитана: shape={df.shape}")

        # Применяем info_query
        if info_query:
            if "columns" in info_query:
                cols = info_query["columns"]
                if all(c in df.columns for c in cols):
                    df = df[cols]
                else:
                    logger.warning(f"Некоторые указанные столбцы отсутствуют: {cols}")
            if "filter_expr" in info_query:
                try:
                    df = df.query(info_query["filter_expr"])
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации: {e}")
            # Можно добавить другие параметры

        # Обрезаем по max_cols и max_rows
        df = df.iloc[:max_rows, :max_cols]

        # Очищаем HTML
        for col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: clean_html_from_cell(x) if pd.notna(x) else x)
            except Exception as e:
                logger.warning(f"Ошибка очистки HTML в столбце {col}: {e}")

        # Формируем summary, если нужно
        summary = ""
        if info_query and info_query.get("summary", False):
            buf = []
            buf.append("Типы столбцов:\n" + str(df.dtypes))
            try:
                buf.append("Статистика (describe):\n" + str(df.describe(include="all")))
            except Exception as e:
                buf.append(f"[Ошибка describe]: {e}")
            summary = "\n".join(buf)

        # Формируем предпросмотр
        preview = df.to_string(index=False)
        result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": preview,
        }
        if summary:
            result["summary"] = summary

        return result

    except Exception as e:
        logger.error(f"analyze_table error: {e}")
        return {"error": str(e)}
