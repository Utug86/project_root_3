import os
import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Set
import signal
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json
import random

from logs import get_logger
from rag_chunk_tracker import ChunkUsageTracker
from rag_retriever import HybridRetriever
from rag_telegram import TelegramPublisher
from rag_lmclient import LMClient
from rag_langchain_tools import enrich_context_with_tools
from rag_prompt_utils import get_prompt_parts
from image_utils import prepare_media_for_post, get_media_type
from utils.config_manager import ConfigManager
from utils.state_manager import StateManager
from utils.exceptions import (
    RAGException,
    ConfigurationError,
    InitializationError,
    ProcessingError,
    ModelError,
    TelegramError,
    FileOperationError,
)

@dataclass
class SystemStats:
    """Статистика работы системы"""
    total_topics: int = 0
    processed_topics: int = 0
    failed_topics: int = 0
    start_time: Optional[datetime] = None
    current_topic: Optional[str] = None
    last_error: Optional[str] = None
    last_processing_time: Optional[float] = None
    total_chars_generated: int = 0
    avg_chars_per_topic: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        stats = asdict(self)
        if self.start_time:
            stats['start_time'] = self.start_time.isoformat()
            stats['running_time'] = str(datetime.now() - self.start_time)
        stats['success_rate'] = (
            (self.processed_topics / self.total_topics * 100)
            if self.total_topics > 0 else 0
        )
        return stats

class RAGSystem:
    def __init__(self):
        # --- Загрузка конфигурации через ConfigManager ---
        self.config_manager = ConfigManager(Path("config/config.json"))
        self.logger = get_logger(__name__, logfile=self.config_manager.get_path("log_dir") / "bot.log")

        # --- Пути из config_manager ---
        self.data_dir = self.config_manager.get_path("data_dir")
        self.log_dir = self.config_manager.get_path("log_dir")
        self.inform_dir = self.config_manager.get_path("inform_dir")
        self.config_dir = Path("config")  # для совместимости, можно убрать если не требуется
        self.media_dir = self.config_manager.get_path("media_dir")
        self.topics_file = self.data_dir / "topics.txt"
        self.processed_topics_file = self.config_manager.get_path("processed_topics_file")
        self.index_file = self.config_manager.get_path("index_file")
        self.context_file = self.config_manager.get_path("context_file")
        self.usage_stats_file = self.config_manager.get_path("usage_stats_file")

        # --- Статистика ---
        self.stats = SystemStats()

        # --- StateManager для прогресса ---
        self.state_manager = StateManager(self.processed_topics_file)

        # --- Флаг для graceful shutdown ---
        self.should_exit = False
        self._register_signals()

    def _register_signals(self):
        # Только SIGINT обрабатывается кроссплатформенно в Python
        signal.signal(signal.SIGINT, self.handle_shutdown)
        # SIGTERM на Windows не поддерживается, но оставим для совместимости
        try:
            signal.signal(signal.SIGTERM, self.handle_shutdown)
        except (AttributeError, ValueError):
            pass

    def setup_paths(self):
        """Проверка и создание необходимых директорий и прав"""
        required_dirs = [
            self.data_dir,
            self.log_dir,
            self.inform_dir,
            self.config_dir,
            self.media_dir,
            self.data_dir / "prompt_1",
            self.data_dir / "prompt_2"
        ]
        for directory in required_dirs:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # Проверка прав на запись
                test_file = directory / ".write_test"
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()
            except Exception as e:
                raise InitializationError(f"Failed to create or write to directory {directory}: {e}")

        # Проверка критичных файлов
        if not self.topics_file.exists():
            raise ConfigurationError("topics.txt not found")
        if not os.access(self.topics_file, os.R_OK):
            raise ConfigurationError(f"topics.txt ({self.topics_file}) is not readable")

    def load_processed_topics(self) -> Set[str]:
        """Загрузка списка обработанных тем через state_manager"""
        try:
            return self.state_manager.get_processed_topics()
        except Exception as e:
            self.logger.critical(f"Failed to load processed topics: {e}")
            raise InitializationError("State file is corrupted or unreadable")

    def save_processed_topic(self, topic: str):
        """Сохранение обработанной темы через state_manager"""
        try:
            self.state_manager.add_processed_topic(topic)
        except Exception as e:
            self.logger.critical(f"Failed to save processed topic: {e}")
            raise FileOperationError("Failed to update state file")

    def add_failed_topic(self, topic: str, error: str):
        """Сохранение темы с ошибкой через state_manager"""
        try:
            self.state_manager.add_failed_topic(topic, error)
        except Exception as e:
            self.logger.critical(f"Failed to add failed topic: {e}")

    async def notify_error(self, message: str):
        """Отправка уведомления об ошибке в Telegram"""
        if hasattr(self, 'telegram'):
            try:
                # TelegramPublisher.send_text - sync, поэтому вызываем через run_in_executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    self.telegram.send_text,
                    f"🚨 RAG System Error:\n{message}"
                )
            except Exception as e:
                self.logger.error(f"Failed to send error notification: {e}")

    def handle_shutdown(self, signum, frame):
        """Обработчик сигналов завершения"""
        self.logger.info("Received shutdown signal, cleaning up...")
        self.should_exit = True

        # Сохранение статистики перед выходом
        try:
            stats = self.stats.to_dict()
            stats_file = self.log_dir / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Statistics saved to {stats_file}")
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {e}")

    def _load_remaining_topics(self) -> list:
        """Загрузка оставшихся тем для обработки"""
        try:
            all_topics = self.topics_file.read_text(encoding='utf-8').splitlines()
            all_topics = [t.strip() for t in all_topics if t.strip()]
            processed = self.load_processed_topics()
            remaining = [t for t in all_topics if t not in processed]
            if not remaining:
                self.logger.warning("No topics left for processing.")
            else:
                self.logger.info(f"Loaded {len(remaining)} remaining topics")
            return remaining
        except Exception as e:
            raise ProcessingError(f"Failed to load topics: {e}")

    async def process_topics(self):
        """Обработка тем из topics.txt"""
        topics = self._load_remaining_topics()
        self.stats.total_topics = len(topics)
        self.stats.start_time = datetime.now()

        if len(topics) == 0:
            self.logger.warning("Topic list is empty, nothing to process.")
            return

        for topic in topics:
            if self.should_exit:
                break
            self.stats.current_topic = topic
            processing_start = datetime.now()
            try:
                self.logger.info(
                    f"Processing topic {self.stats.processed_topics + 1}/{self.stats.total_topics}: {topic}"
                )
                text_length = await self.process_single_topic(topic)
                if text_length is not None and text_length > 0:
                    self.stats.processed_topics += 1
                    self.stats.total_chars_generated += text_length
                    self.stats.avg_chars_per_topic = (
                        self.stats.total_chars_generated / self.stats.processed_topics
                    )
                    self.save_processed_topic(topic)
                    self.stats.last_processing_time = (
                        datetime.now() - processing_start
                    ).total_seconds()
                else:
                    raise ProcessingError("Text length is zero or None")
            except ProcessingError as e:
                error_msg = f"ProcessingError: {e}"
                self.logger.error(error_msg, exc_info=True)
                self.stats.failed_topics += 1
                self.stats.last_error = error_msg
                self.add_failed_topic(topic, error_msg)
                await self.notify_error(error_msg)
                await asyncio.sleep(5)
            except Exception as e:
                error_msg = f"Unexpected error processing topic {topic}: {e}"
                self.logger.critical(error_msg, exc_info=True)
                self.stats.failed_topics += 1
                self.stats.last_error = error_msg
                self.add_failed_topic(topic, error_msg)
                await self.notify_error(error_msg)
                await asyncio.sleep(5)

    async def process_single_topic(self, topic: str) -> Optional[int]:
        """
        Обработка одной темы.
        Возвращает длину сгенерированного текста.
        """
        try:
            # Получение контекста из RAG
            context = self.retriever.retrieve(topic)
            context = enrich_context_with_tools(topic, context, self.inform_dir)

            # Поиск файлов промптов
            prompt1_files = sorted((self.data_dir / "prompt_1").glob("*.txt"))
            prompt2_files = sorted((self.data_dir / "prompt_2").glob("*.txt"))

            if not prompt1_files or not prompt2_files:
                raise ProcessingError("No prompt files found")

            file1 = random.choice(prompt1_files)
            file2 = random.choice(prompt2_files)

            prompt_full = get_prompt_parts(
                data_dir=self.data_dir,
                topic=topic,
                context=context,
                file1=file1,
                file2=file2
            )

            max_chars = (
                self.llm_config["max_chars_with_media"]
                if "{UPLOADFILE}" in prompt_full
                else self.llm_config["max_chars"]
            )

            # Генерация текста (LMClient.generate - должен быть async)
            text = await self.lm.generate(
                topic,
                max_chars=max_chars
            )

            if not text or not isinstance(text, str) or len(text.strip()) == 0:
                raise ProcessingError("Failed to generate text")

            # Отправка в Telegram: методы sync, вызываем через run_in_executor
            loop = asyncio.get_running_loop()
            if "{UPLOADFILE}" in prompt_full:
                await self.handle_media_post(text)
            else:
                # wrap sync call in executor
                result = await loop.run_in_executor(None, self.telegram.send_text, text)
                if result is None:
                    raise ProcessingError("Failed to send text to Telegram")

            self.logger.info(
                f"Successfully processed topic: {topic}, "
                f"text length: {len(text)}"
            )

            return len(text)

        except Exception as e:
            raise ProcessingError(f"Failed to process topic {topic}: {e}")

    async def handle_media_post(self, text: str):
        """Обработка поста с медиафайлом"""
        try:
            media_file = prepare_media_for_post(self.media_dir)
            if not media_file:
                raise ProcessingError("No valid media file found")

            media_type = get_media_type(media_file)
            self.logger.info(f"Selected media file: {media_file} (type: {media_type})")

            loop = asyncio.get_running_loop()
            media_handlers = {
                "image": self.telegram.send_photo,
                "video": self.telegram.send_video,
                "document": self.telegram.send_document,
                "audio": self.telegram.send_audio
            }

            if media_type in media_handlers:
                result = await loop.run_in_executor(
                    None, media_handlers[media_type], media_file, text
                )
                if result is None:
                    raise ProcessingError(f"Failed to send {media_type} to Telegram")
            else:
                self.logger.warning(f"Unknown media type: {media_file}")
                result = await loop.run_in_executor(None, self.telegram.send_text, text)
                if result is None:
                    raise ProcessingError("Failed to send fallback text to Telegram")

        except Exception as e:
            self.logger.error(f"Media handling error: {e}")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.telegram.send_text, text)

    async def run(self):
        """Основной метод запуска системы"""
        try:
            # --- Загрузка параметров из конфиг-менеджера ---
            self.telegram_config = self.config_manager.get_telegram_config()
            self.llm_config = self.config_manager.get_llm_config()
            self.retrieval_config = self.config_manager.get_retrieval_config()
            self.system_config = self.config_manager.get_system_config()

            # --- Инициализация компонентов ---
            self.setup_paths()

            self.usage_tracker = ChunkUsageTracker(
                usage_stats_file=self.usage_stats_file,
                logger=self.logger,
                chunk_usage_limit=self.system_config["chunk_usage_limit"],
                usage_reset_days=self.system_config["usage_reset_days"],
                diversity_boost=self.system_config["diversity_boost"]
            )
            self.usage_tracker.cleanup_old_stats()

            self.retriever = HybridRetriever(
                emb_model=self.retrieval_config["embedding_model"],
                cross_model=self.retrieval_config["cross_encoder"],
                index_file=self.index_file,
                context_file=self.context_file,
                inform_dir=self.inform_dir,
                chunk_size=self.retrieval_config["chunk_size"],
                overlap=self.retrieval_config["overlap"],
                top_k_title=self.retrieval_config["top_k_title"],
                top_k_faiss=self.retrieval_config["top_k_faiss"],
                top_k_final=self.retrieval_config["top_k_final"],
                usage_tracker=self.usage_tracker,
                logger=self.logger
            )

            self.lm = LMClient(
                retriever=self.retriever,
                data_dir=self.data_dir,
                inform_dir=self.inform_dir,
                logger=self.logger,
                model_url=self.llm_config["url"],
                model_name=self.llm_config["model_name"],
                max_tokens=self.llm_config["max_tokens"],
                max_chars=self.llm_config["max_chars"],
                temperature=self.llm_config["temperature"],
                timeout=self.llm_config["timeout"],
                history_lim=self.llm_config["history_limit"],
                system_msg=self.llm_config.get("system_message")
            )

            self.telegram = TelegramPublisher(
                self.telegram_config["bot_token"],
                self.telegram_config["channel_id"],
                logger=self.logger,
                max_retries=self.telegram_config["retry_attempts"],
                retry_delay=self.telegram_config["retry_delay"],
                enable_preview=self.telegram_config["enable_preview"]
            )

            # Проверка соединения с Telegram (sync, обернуть в executor)
            loop = asyncio.get_running_loop()
            telegram_ok = await loop.run_in_executor(None, self.telegram.check_connection)
            if not telegram_ok:
                raise TelegramError("Failed to connect to Telegram")

            self.logger.info("System initialized successfully")

            # Основной цикл обработки
            await self.process_topics()

        except ConfigurationError as e:
            self.logger.critical(f"Configuration error: {e}")
            await self.notify_error(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.critical(f"Unexpected error: {e}", exc_info=True)
            await self.notify_error(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            # Сохранение статистики и очистка
            if hasattr(self, 'usage_tracker'):
                self.usage_tracker.save_statistics()

            # Вывод итоговой статистики
            stats = self.stats.to_dict()
            self.logger.info("Final statistics:")
            for key, value in stats.items():
                self.logger.info(f"{key}: {value}")

            self.logger.info("System shutdown complete")

def main():
    """Точка входа с обработкой всех возможных ошибок"""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    sys.excepthook = handle_exception

    rag_system = RAGSystem()
    try:
        asyncio.run(rag_system.run())
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
