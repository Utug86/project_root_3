from pathlib import Path
import json
from typing import Any, Dict
import logging

class ConfigManager:
    """
    Менеджер конфигурации для всей системы.
    Гарантирует загрузку, валидацию и выдачу параметров с проверкой на ошибки.
    """

    def __init__(self, config_path: Path):
        """
        :param config_path: Путь к json-файлу конфигурации
        """
        self.config_path = config_path
        self.logger = logging.getLogger("config_manager")
        self.config: Dict[str, Any] = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.critical(f"Failed to load configuration: {e}")
            raise

    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        required_sections = ['telegram', 'language_model', 'retrieval', 'system', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

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
        return Path(self.config['paths'][path_key])

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
        Обновление значения в конфиге с сохранением в файл.
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self._save_config()

    def _save_config(self) -> None:
        """Сохранение конфигурации в файл"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
