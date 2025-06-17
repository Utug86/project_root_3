import pandas as pd
from pathlib import Path
from rag_file_utils import clean_html_from_cell
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import os
import re

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
MAX_OUTPUT_ROWS = 30
MAX_OUTPUT_COLS = 10
MAX_MARKDOWN_PREVIEW_CHARS = 3000
CELL_MAX_CHARS = 300
MIN_NONEMPTY_RATIO = 0.1

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

def _sanitize_filename(text: str) -> str:
    """
    Очищает строку для безопасного использования в имени файла.
    """
    safe = re.sub(r"[^\w\-]", "_", str(text))
    return safe[:40] if safe else "empty"

def _is_empty_cell(val: Any) -> bool:
    """
    Проверяет, является ли ячейка пустой (учитывает NaN, None, пустую строку, пустой список/словарь).
    """
    if pd.isna(val):
        return True
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, (list, dict, set)) and len(val) == 0:
        return True
    return False

def _clean_and_reduce_dataframe_with_external_cells(
    df: pd.DataFrame,
    save_dir: Path,
    table_basename: str,
    min_nonempty_ratio: float = MIN_NONEMPTY_RATIO,
    cell_max_chars: int = CELL_MAX_CHARS
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Чистит DataFrame: 
    - удаляет html, 
    - выносит большие ячейки во внешние файлы (безопасно по имени), 
    - удаляет почти пустые столбцы.
    Возвращает новый DataFrame и список описаний вынесенных файлов.
    """
    df = df.copy()
    os.makedirs(save_dir, exist_ok=True)
    external_files_descriptions = []
    col_safe = {col: _sanitize_filename(col) for col in df.columns}

    for col in df.columns:
        for pos, (idx, val) in enumerate(df[col].items()):
            cleaned = "" if _is_empty_cell(val) else clean_html_from_cell(val)
            if isinstance(cleaned, str) and len(cleaned) > cell_max_chars:
                filename = f"{_sanitize_filename(table_basename)}__{col_safe[col]}__row{pos+1}.txt"
                filepath = save_dir / filename
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(cleaned)
                    df.at[idx, col] = f"[Вынесено во внешний файл: {filename}]"
                    external_files_descriptions.append(f"Столбец: '{col}', строка: {pos+1}, файл: {filename}")
                except OSError as e:
                    logger.warning(f"Ошибка при сохранении внешнего файла {filename}: {e}")
                    df.at[idx, col] = "[Ошибка сохранения внешнего файла]"
            else:
                df.at[idx, col] = cleaned

    def column_nonempty_count(series: pd.Series) -> int:
        return (~series.apply(_is_empty_cell)).sum()

    keep_cols = [col for col in df.columns if column_nonempty_count(df[col]) >= min_nonempty_ratio * len(df)]
    df = df[keep_cols]

    # Оставляем только информативные столбцы, удаляем полностью пустые
    # (с учётом всех вариантов пустоты)
    df = df.loc[:, df.apply(lambda col: not all(_is_empty_cell(x) for x in col))]
    return df, external_files_descriptions

def _markdown_table_preview(
    df: pd.DataFrame, 
    max_rows: int = MAX_OUTPUT_ROWS, 
    max_cols: int = MAX_OUTPUT_COLS, 
    max_chars: int = MAX_MARKDOWN_PREVIEW_CHARS
) -> str:
    """
    Формирует markdown-предпросмотр DataFrame с лимитами по строкам, столбцам и символам.
    """
    df_preview = df.iloc[:max_rows, :max_cols]
    lines = []
    colnames = list(df_preview.columns)
    lines.append("| " + " | ".join(colnames) + " |")
    lines.append("|" + "|".join(["---"] * len(colnames)) + "|")
    for _, row in df_preview.iterrows():
        cell_vals = [str(row[col]) if not _is_empty_cell(row[col]) else "" for col in colnames]
        lines.append("| " + " | ".join(cell_vals) + " |")
        if sum(len(l) + 1 for l in lines) > max_chars:
            lines.append(f"| ... (обрезано по лимиту {max_chars} символов) ... |")
            break
    preview = "\n".join(lines)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "\n| ... (обрезано по символам) ... |"
    return preview

def _table_summary(df: pd.DataFrame, max_cols: int = MAX_OUTPUT_COLS) -> str:
    buf = []
    buf.append("**Типы столбцов:**")
    buf.append(str(df.dtypes[:max_cols]))
    try:
        desc = df.describe(include='all').iloc[:, :max_cols]
        buf.append("**Статистика:**")
        buf.append(str(desc))
    except Exception as e:
        buf.append(f"[Ошибка describe]: {e}")
    return "\n".join(buf)

def _format_row(row: pd.Series, colnames: List[str]) -> str:
    """
    Старый формат одной строки таблицы: col: val | col2: val2 ...
    Сохраняется для обратной совместимости (используйте markdown-предпросмотр для новых задач).
    """
    items = []
    for col in colnames:
        val = row[col]
        if _is_empty_cell(val):
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
    max_rows: int = MAX_OUTPUT_ROWS,
    inform_dir: Optional[Path] = None
) -> str:
    """
    Читает и форматирует таблицу для подачи в RAG/LLM. 
    Предпросмотр — markdown-таблица, длинные ячейки выносятся во внешний .txt-файл в папку inform.
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
            except pd.core.computation.ops.UndefinedVariableError as e:
                logger.error(f"Ошибка фильтрации (filter_expr, переменная): {e}")
                return f"[Ошибка фильтрации таблицы]: {e}"
            except pd.core.computation.ops.UndefinedFunctionError as e:
                logger.error(f"Ошибка фильтрации (filter_expr, функция): {e}")
                return f"[Ошибка фильтрации таблицы]: {e}"
            except Exception as e:
                logger.error(f"Ошибка фильтрации (filter_expr): {type(e).__name__}: {e}")
                return f"[Ошибка фильтрации таблицы]: {type(e).__name__}: {e}"

        if df.empty:
            logger.warning("Пустая таблица после фильтрации/чтения")
            return "[Пустая таблица после фильтрации/чтения]"

        save_dir = inform_dir or (file_path.parent / "inform")
        table_basename = file_path.stem
        cleaned_df, external_files = _clean_and_reduce_dataframe_with_external_cells(
            df, save_dir=save_dir, table_basename=table_basename
        )

        preview_md = _markdown_table_preview(cleaned_df, max_rows=max_rows, max_cols=MAX_OUTPUT_COLS)
        summary = ""
        if len(df) > max_rows:
            summary = "\n\n" + _table_summary(df)

        external_note = ""
        if external_files:
            external_note = (
                "\n\n**Внимание:** Некоторые ячейки были слишком длинными и вынесены во внешние файлы в папку `inform`.\n"
                + "\n".join(f"- {descr}" for descr in external_files)
            )

        result = preview_md
        if summary:
            result += "\n\n" + summary
        if external_note:
            result += external_note
        return result
    except Exception as e:
        logger.error(f"process_table_for_rag error: {type(e).__name__}: {e}")
        return f"[Ошибка обработки таблицы для RAG]: {type(e).__name__}: {e}"

def analyze_table(
    table_path: Path,
    info_query: Optional[dict] = None,
    max_rows: int = 18,
    max_cols: int = 10,
    inform_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Анализирует табличный файл: возвращает preview (markdown), типы, статистику.
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
                except pd.core.computation.ops.UndefinedVariableError as e:
                    logger.warning(f"Ошибка фильтрации (переменная): {e}")
                except pd.core.computation.ops.UndefinedFunctionError as e:
                    logger.warning(f"Ошибка фильтрации (функция): {e}")
                except Exception as e:
                    logger.warning(f"Ошибка фильтрации: {type(e).__name__}: {e}")

        save_dir = inform_dir or (table_path.parent / "inform")
        table_basename = table_path.stem
        cleaned_df, external_files = _clean_and_reduce_dataframe_with_external_cells(
            df, save_dir=save_dir, table_basename=table_basename
        )

        df_preview = cleaned_df.iloc[:max_rows, :max_cols]

        preview = _markdown_table_preview(
            df_preview, max_rows=max_rows, max_cols=max_cols, max_chars=MAX_MARKDOWN_PREVIEW_CHARS
        )
        result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "preview": preview,
        }
        if info_query and info_query.get("summary", False):
            result["summary"] = _table_summary(df, max_cols=max_cols)
        if external_files:
            result["external_files"] = external_files
            result["external_note"] = (
                f"Длинные ячейки вынесены в директорию {save_dir.name}:\n" + "\n".join(external_files)
            )
        return result

    except Exception as e:
        logger.error(f"analyze_table error: {type(e).__name__}: {e}")
        return {"error": f"{type(e).__name__}: {e}"}
