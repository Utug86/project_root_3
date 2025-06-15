import logging
import os
from typing import Optional, Union

def get_logger(
    name: str,
    logfile: Optional[Union[str, os.PathLike]] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    silent: bool = False,
    rolling: bool = False,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    file_level: Optional[int] = None,
    stream_level: Optional[int] = None,
) -> logging.Logger:
    """
    Создает и возвращает логгер с заданным именем.
    - logfile: путь к лог-файлу. Если не указан — лог только в консоль.
    - level: основной уровень логгирования (по умолчанию INFO).
    - log_format: формат лог-сообщений (по умолчанию стандартный формат).
    - silent: не выводить логи в консоль (только файл, если задан).
    - rolling: использовать RotatingFileHandler вместо обычного FileHandler.
    - max_bytes: максимальный размер лога для ротации (если rolling=True).
    - backup_count: сколько файлов хранить при ротации.
    - file_level / stream_level: уровни логирования для файла и консоли (по отдельности).
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if not logger.hasHandlers():
        if not log_format:
            log_format = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
        formatter = logging.Formatter(log_format)

        # Файл-логгер
        if logfile:
            log_dir = os.path.dirname(str(logfile))
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except Exception as e:
                    raise RuntimeError(f"Не удалось создать директорию для логов: {log_dir}: {e}")
            if rolling:
                from logging.handlers import RotatingFileHandler
                fh = RotatingFileHandler(
                    logfile, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
            else:
                fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(file_level if file_level is not None else level)
            logger.addHandler(fh)

        # Консоль-логгер
        if not silent:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            sh.setLevel(stream_level if stream_level is not None else level)
            logger.addHandler(sh)

        logger.setLevel(level)
    else:
        # Логируем, если get_logger вызывается повторно с другими параметрами
        logger.debug(
            f"get_logger('{name}', ...): логгер уже инициализирован, новые параметры проигнорированы."
        )

    return logger
