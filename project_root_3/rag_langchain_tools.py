import logging
from utils.search_utils import web_search
from utils.rag_text_utils import safe_eval
from utils.rag_table_utils import analyze_table
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("rag_langchain_tools")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [rag_langchain_tools] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

TOOL_KEYWORDS = {
    "web": ["найди", "поиск", "интернет", "lookup", "search", "google", "bing", "duckduckgo"],
    "calc": ["выгод", "посчит", "calculate", "profit", "выбери", "сколько", "рассчитай"],
    "table": ["таблиц", "excel", "csv", "xlsx", "анализируй", "данные", "отчет", "таблица"]
}

def tool_internet_search(query: str, num_results: int = 8) -> str:
    """
    Выполняет интернет-поиск по запросу.
    """
    logger.info(f"Вызов интернет-поиска по запросу: {query}")
    results = web_search(query, num_results=num_results)
    if not results:
        logger.warning("Интернет-поиск не дал результатов")
        return "[Интернет-поиск не дал результатов]"
    return "\n".join(results)

def tool_calculator(expr: str, variables: Optional[Dict[str, Any]] = None) -> str:
    """
    Выполняет безопасный расчет выражения.
    """
    logger.info(f"Вызов калькулятора с выражением: {expr}")
    try:
        return str(safe_eval(expr, variables=variables))
    except Exception as e:
        logger.error(f"Ошибка калькуляции: {e}")
        return f"[Ошибка калькуляции]: {e}"

def tool_table_analysis(
    table_filename: str,
    info_query: Optional[dict]=None,
    inform_dir: Optional[str]=None,
    max_rows: int = 18,
    max_cols: int = 10
) -> str:
    """
    Анализирует таблицу по запросу.
    """
    logger.info(f"Анализ таблицы: {table_filename}")
    try:
        file_path = Path(inform_dir) / table_filename
        return analyze_table(file_path, info_query, max_rows=max_rows, max_cols=max_cols)
    except Exception as e:
        logger.error(f"Ошибка анализа таблицы: {e}")
        return f"[Ошибка анализа таблицы]: {e}"

# ... все ваши импорты и TOOL_KEYWORDS ...

def smart_tool_selector(
    topic: str,
    context: str,
    inform_dir: str,
    tool_keywords: Optional[Dict[str, List[str]]] = None,
    tool_log: Optional[List[str]] = None,
    max_tool_results: int = 8,
    enforce: bool = False
) -> str:
    tool_keywords = tool_keywords or TOOL_KEYWORDS
    tool_log = tool_log or []
    topic_lc = topic.lower()
    results = []
    used_tools = []

    # Web search
    if any(x in topic_lc for x in tool_keywords["web"]):
        logger.info("[smart_tool_selector] Web search triggered")
        tool_log.append("web_search")
        results.append("[Интернет]:\n" + tool_internet_search(topic, num_results=max_tool_results))
        used_tools.append("web_search")
    # Calculator
    if any(x in topic_lc for x in tool_keywords["calc"]):
        import re
        logger.info("[smart_tool_selector] Calculator triggered")
        tool_log.append("calculator")
        m = re.search(r"(посчитай|calculate|выгоднее|выгодность|сколько)[^\d]*(.+)", topic_lc)
        expr = m.group(2) if m else topic
        results.append("[Калькулятор]:\n" + tool_calculator(expr))
        used_tools.append("calculator")
    # Table
    if any(x in topic_lc for x in tool_keywords["table"]):
        logger.info("[smart_tool_selector] Table analysis triggered")
        tool_log.append("analyze_table")
        table_files = [f.name for f in Path(inform_dir).glob("*.csv")] + [f.name for f in Path(inform_dir).glob("*.xlsx")]
        if table_files:
            results.append("[Таблица]:\n" + tool_table_analysis(table_files[0], None, inform_dir))
            used_tools.append("analyze_table")
        else:
            results.append("[Нет подходящих таблиц для анализа]")

    # Fallback если ничего не найдено, но enforce
    if not results and enforce:
        logger.info("[smart_tool_selector] Enforce-флаг активен: вызываем fallback инструмент")
        fallback_result = "[RAG: инструментальное расширение не найдено, добавлен базовый интернет-поиск]\n"
        fallback_result += tool_internet_search(topic, num_results=1)
        tool_log.append("fallback_web_search")
        results.append(fallback_result)
        used_tools.append("fallback_web_search")

    if used_tools:
        logger.info(f"Вызваны инструменты: {used_tools}")
    if results:
        return "\n\n".join(results)
    else:
        logger.info("Ни один инструмент не был вызван")
        return ""

def enrich_context_with_tools(
    topic: str,
    context: str,
    inform_dir: str,
    max_tool_results: int = 8,
    enforce: bool = False
) -> str:
    logger.info("Расширение контекста инструментами...")
    tool_result = smart_tool_selector(
        topic, context, inform_dir, max_tool_results=max_tool_results, enforce=enforce
    )
    if tool_result:
        context = context + "\n\n[Инструментальное расширение]:\n" + tool_result
        logger.info("Контекст расширен инструментами.")
    else:
        logger.info("Инструменты не были использованы для расширения контекста.")
    return context
