from pathlib import Path
import json
from typing import Any, Dict, Optional
import logging
import shutil

class ConfigCorruptionError(Exception):
    """Явная критическая ошибка при повреждении конфигурационного файла"""
    pass

class ConfigManager:
    """
    Менеджер конфигурации для всей системы.
    Гарантирует загрузку, резервирование, atomic сохранение, валидацию структуры и выдачу параметров.
    """

    REQUIRED_SECTIONS = ['telegram', 'language_model', 'retrieval', 'system', 'paths']
    # Ожидаемая структура для каждой секции (можно дополнить по необходимости)
    REQUIRED_KEYS = {
        'telegram': ['bot_token_file', 'channel_id_file', 'retry_attempts', 'retry_delay', 'enable_preview'],
        'language_model': ['url', 'model_name', 'max_tokens', 'max_chars', 'max_chars_with_media', 'temperature', 'timeout', 'history_limit'],
        'retrieval': ['embedding_model', 'cross_encoder', 'chunk_size', 'overlap', 'top_k_title', 'top_k_faiss', 'top_k_final'],
        'system': ['chunk_usage_limit', 'usage_reset_days', 'diversity_boost'],
        'paths': ['log_dir', 'data_dir', 'inform_dir', 'media_dir', 'processed_topics_file', 'index_file', 'context_file', 'usage_stats_file']
    }

    def __init__(self, config_path: Path, logger: Optional[logging.Logger] = None):
        """
        :param config_path: Путь к json-файлу конфигурации
        """
        self.config_path = Path(config_path)
        self.logger = logger or self._default_logger()
        self.config: Dict[str, Any] = self._try_load_config_or_raise()
        self._validate_config()

    def _default_logger(self) -> logging.Logger:
        logger = logging.getLogger("config_manager")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] [config_manager] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _safe_backup(self, source: Path, suffix: str = ".bak") -> Optional[Path]:
        """
        Создать резервную копию файла с уникальным суффиксом (timestamp).
        """
        try:
            if not source.exists():
                return None
            ts = __import__('datetime').datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source.stem}{suffix}_{ts}{source.suffix}"
            backup_path = source.parent / backup_name
            shutil.copy2(source, backup_path)
            self.logger.warning(f"Config backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to backup config file: {e}")
            return None

    def _try_load_config_or_raise(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла, резервирование битых файлов, явный сброс при ошибке."""
        if not self.config_path.exists():
            raise ConfigCorruptionError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            self.logger.critical(f"Failed to load configuration: {e}")
            self._safe_backup(self.config_path, suffix=".broken")
            raise ConfigCorruptionError(f"Config file '{self.config_path}' is corrupted: {e}. Backup created. Manual intervention required.")

    def _validate_config(self) -> None:
        """Валидация структуры конфигурации (разделы + ключи)"""
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
            expected_keys = self.REQUIRED_KEYS.get(section, [])
            for key in expected_keys:
                if key not in self.config[section]:
                    raise ValueError(f"Missing key '{key}' in section '{section}' of config")
        self.logger.info("Config structure validated")

    def get_telegram_config(self) -> Dict[str, Any]:
        """
        Получение и проверка конфигурации Telegram.
        :return: словарь с ключами 'bot_token', 'channel_id' и настройками Telegram
        :raises: ValueError, если файлы токена или channel_id не найдены
        """
        config = self.config['telegram'].copy()
        try:
            token_file = Path(config['bot_token_file'])
            channel_file = Path(config['channel_id_file'])

            if not token_file.exists():
                raise ValueError(f"Telegram token file not found: {token_file}")
            if not channel_file.exists():
                raise ValueError(f"Channel ID file not found: {channel_file}")

            config['bot_token'] = token_file.read_text(encoding='utf-8').strip()
            config['channel_id'] = channel_file.read_text(encoding='utf-8').strip()
        except Exception as e:
            self.logger.critical(f"Failed to load Telegram credentials: {e}")
            raise
        return config

    def get_llm_config(self) -> Dict[str, Any]:
        """Получение настроек LLM"""
        return self.config['language_model']

    def get_retrieval_config(self) -> Dict[str, Any]:
        """Получение retrieval-конфига"""
        return self.config['retrieval']

    def get_system_config(self) -> Dict[str, Any]:
        """Получение системных параметров"""
        return self.config['system']

    def get_path(self, path_key: str) -> Path:
        """
        Получение пути из конфигурации.
        :param path_key: Ключ из секции 'paths'
        :return: Path-объект
        :raises KeyError: если ключ не найден
        """
        if path_key not in self.config['paths']:
            raise KeyError(f"Path not found in config: {path_key}")
        path = Path(self.config['paths'][path_key])
        # Проверка существования и прав доступа (чтение/запись)
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise FileNotFoundError(f"Directory for path '{path}' could not be created: {e}")
        return path

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Получение значения по секции и ключу.
        :param section: Имя секции (str)
        :param key: Имя ключа (str)
        :param default: Значение по умолчанию
        """
        return self.config.get(section, {}).get(key, default)

    def update(self, section: str, key: str, value: Any) -> None:
        """
        Обновление значения в конфиге с atomic сохранением в файл.
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self._atomic_save_config()

    def _atomic_save_config(self) -> None:
        """Atomic-сохранение конфигурации в файл с резервным копированием."""
        tmp_file = self.config_path.with_suffix('.tmp')
        try:
            # Сохраняем старый конфиг
            if self.config_path.exists():
                self._safe_backup(self.config_path, suffix=".prev")
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            tmp_file.replace(self.config_path)
            self.logger.info(f"Config saved atomically to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise

    def print_config_structure(self) -> None:
        """
        Печать структуры ожидаемого конфига (для документации и отладки).
        """
        import pprint
        expected = {section: self.REQUIRED_KEYS[section] for section in self.REQUIRED_SECTIONS}
        pprint.pprint(expected)

    def reload_config(self) -> None:
        """Принудительно перечитать конфиг из файла (например, если он был изменен извне)"""
        self.config = self._try_load_config_or_raise()
        self._validate_config()
