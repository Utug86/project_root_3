class RAGException(Exception):
    """Базовый класс для исключений RAG системы"""
    pass

class ConfigurationError(RAGException):
    """Ошибки конфигурации"""
    pass

class InitializationError(RAGException):
    """Ошибки инициализации компонентов"""
    pass

class ProcessingError(RAGException):
    """Ошибки обработки данных"""
    pass

class ModelError(RAGException):
    """Ошибки языковой модели"""
    pass

class TelegramError(RAGException):
    """Ошибки взаимодействия с Telegram"""
    pass

class FileOperationError(RAGException):
    """Ошибки файловых операций"""
    pass
