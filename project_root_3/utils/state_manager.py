import json
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Set, Optional, Union
from datetime import datetime
import time

class StateManager:
    """
    Устойчивый менеджер состояния для отслеживания прогресса обработки тем.
    Гарантирует recoverability, атомарность, защиту от гонок, автоматическое резервирование, валидацию структуры, масштабируемость.
    """

    _file_lock = threading.Lock()
    _required_keys = {
        "last_update",
        "processed_topics",
        "failed_topics",
        "statistics"
    }
    _required_stats_keys = {
        "total_processed",
        "successful",
        "failed"
    }

    def __init__(
        self,
        state_file: Union[str, Path],
        backup_dir: Optional[Union[str, Path]] = None,
        logger: Optional[logging.Logger] = None,
        autosave: bool = True,
    ):
        """
        :param state_file: Путь к файлу состояния (JSON).
        :param backup_dir: Путь для хранения резервных копий (по умолчанию — рядом с state_file).
        :param logger: Logger экземпляр, если не задан — создается свой.
        :param autosave: Автоматически сохранять состояние после изменений.
        """
        self.state_file = Path(state_file)
        self.backup_dir = Path(backup_dir) if backup_dir else self.state_file.parent
        self.autosave = autosave
        self.logger = logger or self._default_logger()
        self.state: Dict[str, Any] = self._load_state()

    def _default_logger(self) -> logging.Logger:
        logger = logging.getLogger("state_manager")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] [state_manager] %(message)s')
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
            ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source.stem}{suffix}_{ts}{source.suffix}"
            backup_path = self.backup_dir / backup_name
            shutil.copy2(source, backup_path)
            self.logger.warning(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to backup state file: {e}")
            return None

    def _validate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Гарантирует наличие и корректность всех ключей и структур.
        """
        # Основные ключи
        for k in self._required_keys:
            if k not in state:
                self.logger.warning(f"Key '{k}' missing in state, initializing default.")
                if k == "last_update":
                    state[k] = datetime.utcnow().isoformat()
                elif k == "processed_topics":
                    state[k] = set()
                elif k == "failed_topics":
                    state[k] = {}
                elif k == "statistics":
                    state[k] = dict.fromkeys(self._required_stats_keys, 0)
        # processed_topics — set
        if isinstance(state["processed_topics"], list):
            state["processed_topics"] = set(state["processed_topics"])
        elif not isinstance(state["processed_topics"], set):
            state["processed_topics"] = set()
        # failed_topics — dict
        if not isinstance(state["failed_topics"], dict):
            state["failed_topics"] = {}
        # statistics — dict с ключами
        if not isinstance(state["statistics"], dict):
            state["statistics"] = dict.fromkeys(self._required_stats_keys, 0)
        for k in self._required_stats_keys:
            if k not in state["statistics"]:
                state["statistics"][k] = 0
        return state

    def _load_state(self) -> Dict[str, Any]:
        """
        Надежная загрузка состояния: восстановление структуры, резервирование битых файлов, восстановление из tmp при сбое.
        """
        with self._file_lock:
            # Если остался tmp-файл (сбой при сохранении) — восстановить из него
            tmp_file = self.state_file.with_suffix('.tmp')
            if tmp_file.exists():
                self.logger.error(f"Detected leftover tmp state file: {tmp_file}. Attempting recovery.")
                self._safe_backup(self.state_file, suffix=".corrupted")
                try:
                    tmp_file.replace(self.state_file)
                    self.logger.info(f"Recovered state from tmp file: {tmp_file}")
                except Exception as e:
                    self.logger.critical(f"Failed to recover from tmp state: {e}")

            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data = self._validate_state(data)
                    return data
                except Exception as e:
                    # Сохраняем поврежденный файл
                    self.logger.error(f"Failed to load state: {e}, backing up and creating default state.")
                    self._safe_backup(self.state_file, suffix=".broken")
                    return self._create_default_state()
            else:
                self.logger.info("No state file found, creating new default state.")
                return self._create_default_state()

    def _create_default_state(self) -> Dict[str, Any]:
        """Создание состояния по умолчанию."""
        return {
            "last_update": datetime.utcnow().isoformat(),
            "processed_topics": set(),
            "failed_topics": {},
            "statistics": {
                "total_processed": 0,
                "successful": 0,
                "failed": 0
            }
        }

    def save_state(self) -> None:
        """
        Атомарно сохраняет состояние с резервным копированием и защитой от гонок.
        """
        with self._file_lock:
            # Валидируем структуру перед записью
            state_copy = self._validate_state(self.state.copy())
            state_copy["processed_topics"] = list(state_copy["processed_topics"])
            state_copy["last_update"] = datetime.utcnow().isoformat()
            tmp_file = self.state_file.with_suffix('.tmp')
            try:
                # Бэкап основного файла перед записью
                if self.state_file.exists():
                    self._safe_backup(self.state_file, suffix=".prev")
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump(state_copy, f, indent=4, ensure_ascii=False)
                tmp_file.replace(self.state_file)
            except Exception as e:
                self.logger.critical(f"Failed to save state: {e}")
                raise

    def add_processed_topic(self, topic: str) -> None:
        """
        Добавить обработанную тему. Не увеличивает статистику, если уже обработана.
        """
        with self._file_lock:
            if topic not in self.state["processed_topics"]:
                self.state["processed_topics"].add(topic)
                self.state["statistics"]["total_processed"] += 1
                self.state["statistics"]["successful"] += 1
                if self.autosave:
                    self.save_state()
            else:
                self.logger.info(f"Topic already processed: {topic}")

    def add_failed_topic(self, topic: str, error: str, max_attempts: Optional[int] = None) -> None:
        """
        Добавить тему с ошибкой. Если max_attempts задан и превышен — не добавлять.
        """
        with self._file_lock:
            entry = self.state["failed_topics"].get(topic, {})
            attempts = entry.get("attempts", 0) + 1
            if max_attempts is not None and attempts > max_attempts:
                self.logger.warning(f"Topic {topic} exceeded max_attempts={max_attempts}, skipping.")
                return
            self.state["failed_topics"][topic] = {
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                "attempts": attempts
            }
            self.state["statistics"]["failed"] += 1
            if self.autosave:
                self.save_state()

    def get_processed_topics(self) -> Set[str]:
        """Получить копию множества обработанных тем (read-only)."""
        with self._file_lock:
            return set(self.state["processed_topics"])

    def get_failed_topics(self) -> Dict[str, Dict[str, Any]]:
        """Получить копию dict тем с ошибками (read-only)."""
        with self._file_lock:
            return {k: v.copy() for k, v in self.state["failed_topics"].items()}

    def get_statistics(self) -> Dict[str, int]:
        """Получить копию статистики (read-only)."""
        with self._file_lock:
            return dict(self.state["statistics"])

    def clear_failed_topics(self) -> None:
        """Очистить список тем с ошибками."""
        with self._file_lock:
            self.state["failed_topics"].clear()
            if self.autosave:
                self.save_state()

    def batch_add_processed(self, topics: Set[str]) -> None:
        """
        Групповое добавление обработанных тем.
        """
        with self._file_lock:
            new_topics = topics - self.state["processed_topics"]
            self.state["processed_topics"].update(new_topics)
            self.state["statistics"]["total_processed"] += len(new_topics)
            self.state["statistics"]["successful"] += len(new_topics)
            if self.autosave:
                self.save_state()

    def batch_add_failed(self, topics_errors: Dict[str, str], max_attempts: Optional[int] = None) -> None:
        """
        Групповое добавление тем с ошибками.
        """
        with self._file_lock:
            for topic, error in topics_errors.items():
                entry = self.state["failed_topics"].get(topic, {})
                attempts = entry.get("attempts", 0) + 1
                if max_attempts is not None and attempts > max_attempts:
                    self.logger.warning(f"Topic {topic} exceeded max_attempts={max_attempts}, skipping.")
                    continue
                self.state["failed_topics"][topic] = {
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                    "attempts": attempts
                }
                self.state["statistics"]["failed"] += 1
            if self.autosave:
                self.save_state()

    def reload_state(self) -> None:
        """
        Принудительно перечитать состояние из файла (например, если другая нода изменила state).
        """
        with self._file_lock:
            self.state = self._load_state()

    def manual_save(self) -> None:
        """
        Принудительно сохранить состояние, если autosave=False.
        """
        self.save_state()

    def to_json(self) -> str:
        """
        Получить сериализованное состояние в виде строки (только для диагностики).
        """
        with self._file_lock:
            state_copy = self._validate_state(self.state.copy())
            state_copy["processed_topics"] = list(state_copy["processed_topics"])
            return json.dumps(state_copy, ensure_ascii=False, indent=2)

    def from_json(self, json_str: str) -> None:
        """
        Загрузить состояние из строки (например, для тестов или восстановления).
        """
        with self._file_lock:
            try:
                data = json.loads(json_str)
                self.state = self._validate_state(data)
                if self.autosave:
                    self.save_state()
            except Exception as e:
                self.logger.error(f"Failed to load state from json string: {e}")

    # Advanced: поддержка контекстного менеджера для гарантированного сохранения
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.autosave:
            self.save_state()
