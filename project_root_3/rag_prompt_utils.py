from pathlib import Path
from typing import Optional, Union, Dict, Any
from logs import get_logger
import random
import re

logger = get_logger("rag_prompt_utils")

class PromptTemplateCache:
    _cache: Dict[str, str] = {}
    @classmethod
    def get(cls, path: Path) -> Optional[str]:
        key = str(path.resolve())
        if key in cls._cache:
            return cls._cache[key]
        if not path.exists():
            logger.warning(f"Шаблон промпта не найден: {path}")
            return None
        try:
            text = path.read_text(encoding="utf-8")
            cls._cache[key] = text
            return text
        except Exception as e:
            logger.error(f"Ошибка чтения шаблона: {path}: {e}")
            return None

def safe_format(template: str, variables: Dict[str, Any]) -> str:
    def repl(match):
        var = match.group(1)
        return str(variables.get(var, f"{{{var}}}"))
    return re.sub(r"\{(\w+)\}", repl, template)

def get_prompt_parts(
    data_dir: Union[str, Path],
    topic: str,
    context: str,
    uploadfile: Optional[Union[str, Path]] = None,
    file1: Optional[Union[str, Path]] = None,
    file2: Optional[Union[str, Path]] = None,
    extra_vars: Optional[Dict[str, Any]] = None,
    max_context_len_upload: int = 1024,
    max_context_len_no_upload: int = 4096,
    max_prompt_len: int = 8192,
) -> str:
    """
    Составляет промпт для LLM на основе шаблонов и переданных параметров.
    Условия:
      - {TOPIC}: каждая строка из topics.txt, подставляется отдельно
      - {CONTEXT}: весь материал из RAG+интернета, лимит 4096 или 1024 если есть {UPLOADFILE}
      - {UPLOADFILE}: имя рандомного файла из media или статус ошибки
    """
    data_dir = Path(data_dir)
    if file1 is not None:
        file1 = Path(file1)
    if file2 is not None:
        file2 = Path(file2)
    if uploadfile is not None:
        uploadfile_path = Path(uploadfile)
    else:
        uploadfile_path = None

    # Проверка существования data_dir
    if not data_dir.exists() or not data_dir.is_dir():
        logger.error(f"data_dir '{data_dir}' не существует или не является директорией")
        return f"{topic}\n\n{context[:max_context_len_no_upload]}"

    def read_template(path: Path) -> Optional[str]:
        return PromptTemplateCache.get(path)

    prompt1_dir = data_dir / "prompt_1"
    prompt2_dir = data_dir / "prompt_2"
    template = None

    # Детерминированный шаблон
    if file1 is not None and file2 is not None and file1.exists() and file2.exists():
        logger.info(f"Детерминированный шаблон: {file1.name} + {file2.name}")
        txt1 = read_template(file1)
        txt2 = read_template(file2)
        if txt1 is not None and txt2 is not None:
            template = txt1 + "\n" + txt2
    # Случайные шаблоны
    elif prompt1_dir.exists() and prompt2_dir.exists():
        prompt1_files = list(prompt1_dir.glob("*.txt"))
        prompt2_files = list(prompt2_dir.glob("*.txt"))
        if prompt1_files and prompt2_files:
            f1 = random.choice(prompt1_files)
            f2 = random.choice(prompt2_files)
            logger.info(f"Случайный шаблон: {f1.name} + {f2.name}")
            txt1 = read_template(f1)
            txt2 = read_template(f2)
            if txt1 is not None and txt2 is not None:
                template = txt1 + "\n" + txt2

    # Fallback
    if template is None:
        prompt_file = data_dir / "prompt.txt"
        if prompt_file.exists():
            logger.warning("Fallback на prompt.txt")
            template = read_template(prompt_file)
        else:
            logger.warning("Fallback на plain topic + context")
            return f"{topic}\n\n{context[:max_context_len_no_upload]}"

    if template is None or not template.strip():
        logger.error("Шаблон пустой или не удалось прочитать, возврат plain topic + context")
        return f"{topic}\n\n{context[:max_context_len_no_upload]}"

    # Определяем лимит context
    has_uploadfile = "{UPLOADFILE}" in template
    if has_uploadfile:
        context = context[:max_context_len_upload]
    else:
        context = context[:max_context_len_no_upload]

    # Формируем переменные для шаблона
    variables = {
        "TOPIC": topic,
        "CONTEXT": context,
    }
    if extra_vars:
        variables.update(extra_vars)
    # uploadfile logic
    if has_uploadfile:
        if uploadfile_path is not None:
            try:
                if uploadfile_path.exists():
                    variables["UPLOADFILE"] = uploadfile_path.name
                else:
                    variables["UPLOADFILE"] = f"[Файл не найден: {uploadfile_path.name}]"
            except Exception as e:
                variables["UPLOADFILE"] = "[Ошибка с файлом]"
                logger.error(f"Ошибка обработки uploadfile: {e}")
        else:
            variables["UPLOADFILE"] = "[Файл не передан]"

    prompt_out = safe_format(template, variables).strip()
    if len(prompt_out) > max_prompt_len:
        logger.warning(f"Промпт превышает лимит {max_prompt_len}, будет обрезан.")
        prompt_out = prompt_out[:max_prompt_len-10] + "..."
    return prompt_out
