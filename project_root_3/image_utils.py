import random
from pathlib import Path
from typing import Optional, Tuple, List, Set, Union
from PIL import Image, UnidentifiedImageError
import logging
import mimetypes
from enum import Enum, auto

logger = logging.getLogger("image_utils")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [image_utils] %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SUPPORTED_IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".mkv"}
SUPPORTED_DOC_EXTS: Set[str] = {".pdf", ".docx", ".doc", ".txt", ".csv", ".xlsx"}
SUPPORTED_AUDIO_EXTS: Set[str] = {".mp3", ".wav", ".ogg"}
SUPPORTED_MEDIA_EXTS: Set[str] = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS | SUPPORTED_DOC_EXTS | SUPPORTED_AUDIO_EXTS

MAX_IMAGE_SIZE: Tuple[int, int] = (1280, 1280)
MAX_FILE_SIZE_MB: int = 50

class MediaValidationStatus(Enum):
    OK = auto()
    NOT_FOUND = auto()
    SYMLINK = auto()
    OUT_OF_DIR = auto()
    NOT_SUPPORTED = auto()
    TOO_LARGE = auto()
    CANNOT_OPEN = auto()
    UNKNOWN = auto()

class MediaValidationResult:
    def __init__(self, valid: bool, status: MediaValidationStatus, path: Optional[Path], message: str = ""):
        self.valid = valid
        self.status = status
        self.path = path
        self.message = message

    def __bool__(self):
        return self.valid

    def __repr__(self):
        return f"<MediaValidationResult status={self.status} path={self.path} msg={self.message}>"

def is_safe_media_path(path: Path, media_dir: Path) -> bool:
    """
    Проверяет, что path лежит строго внутри media_dir.
    Защита от directory traversal.
    """
    try:
        # Python >=3.9
        return path.resolve().is_relative_to(media_dir.resolve())
    except AttributeError:  # Python <3.9
        try:
            return str(path.resolve().as_posix()).startswith(media_dir.resolve().as_posix() + "/")
        except Exception as e:
            logger.error(f"Ошибка проверки пути: {e}")
            return False
    except Exception as e:
        logger.error(f"Ошибка проверки пути: {e}")
        return False

def pick_random_media_file(media_dir: Path, allowed_exts: Optional[Set[str]] = None, max_retries: int = 5) -> Optional[Path]:
    """
    Случайно выбирает валидный файл из media_dir (рекурсивно) с поддерживаемым расширением.
    Делает max_retries попыток выбрать валидный файл.
    """
    allowed_exts = allowed_exts or SUPPORTED_MEDIA_EXTS
    files = [f for f in media_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]
    if not files:
        logger.warning(f"Нет файлов с поддерживаемыми расширениями в {media_dir}")
        return None
    attempt = 0
    while attempt < max_retries and files:
        file = random.choice(files)
        result = validate_media_file(file, media_dir)
        if result.valid:
            return file
        else:
            logger.info(f"Пропущен невалидный файл при попытке выбора: {file} ({result.status})")
            files.remove(file)
            attempt += 1
    logger.warning("Не удалось выбрать валидный файл после повторных попыток.")
    return None

def validate_media_file(path: Path, media_dir: Path = Path("media")) -> MediaValidationResult:
    """
    Проверяет валидность файла:
    - В media_dir
    - Поддерживаемое расширение
    - Не превышает лимит размера
    - Существует
    - Не symlink
    """
    if not path.exists():
        logger.error(f"Файл не найден: {path}")
        return MediaValidationResult(False, MediaValidationStatus.NOT_FOUND, path, "Файл не найден")
    if path.is_symlink():
        logger.error(f"Файл является symlink: {path}")
        return MediaValidationResult(False, MediaValidationStatus.SYMLINK, path, "Файл является symlink")
    if not is_safe_media_path(path, media_dir):
        logger.error(f"Файл вне папки media: {path}")
        return MediaValidationResult(False, MediaValidationStatus.OUT_OF_DIR, path, "Файл вне папки media")
    if path.suffix.lower() not in SUPPORTED_MEDIA_EXTS:
        logger.error(f"Неподдерживаемый формат: {path.suffix} ({path})")
        return MediaValidationResult(False, MediaValidationStatus.NOT_SUPPORTED, path, f"Неподдерживаемый формат: {path.suffix}")
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        logger.error(f"Файл слишком большой (> {MAX_FILE_SIZE_MB} МБ): {path}")
        return MediaValidationResult(False, MediaValidationStatus.TOO_LARGE, path, f"Файл слишком большой (>{MAX_FILE_SIZE_MB} МБ)")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        logger.warning(f"Не удалось определить MIME-тип: {path}")
    return MediaValidationResult(True, MediaValidationStatus.OK, path, "OK")

def get_media_type(path: Path) -> str:
    """
    Определяет тип файла по расширению.
    """
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTS:
        return "image"
    elif ext in SUPPORTED_VIDEO_EXTS:
        return "video"
    elif ext in SUPPORTED_DOC_EXTS:
        return "document"
    elif ext in SUPPORTED_AUDIO_EXTS:
        return "audio"
    return "unknown"

def process_image(path: Path, output_dir: Optional[Path] = None, max_size: Tuple[int, int] = MAX_IMAGE_SIZE) -> Optional[Path]:
    """
    Уменьшает изображение до max_size, если требуется. Возвращает путь к новому файлу.
    """
    try:
        with Image.open(path) as img:
            if img.size[0] <= max_size[0] and img.size[1] <= max_size[1]:
                logger.info(f"Изображение уже в допустимом размере: {path}")
                return path
            img.thumbnail(max_size, Image.LANCZOS)
            out_dir = output_dir or path.parent
            out_path = out_dir / f"{path.stem}_resized{path.suffix}"
            img.save(out_path)
            logger.info(f"Изображение уменьшено и сохранено в: {out_path}")
            return out_path
    except UnidentifiedImageError:
        logger.error(f"Не удалось открыть изображение: {path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка обработки изображения {path}: {e}")
        return None

def get_all_media_files(media_dir: Path, allowed_exts: Optional[Set[str]] = None) -> List[Path]:
    """
    Список всех файлов в media_dir c поддерживаемыми расширениями.
    """
    allowed_exts = allowed_exts or SUPPORTED_MEDIA_EXTS
    return [f for f in media_dir.rglob("*") if f.is_file() and f.suffix.lower() in allowed_exts]

def prepare_media_for_post(media_dir: Path = Path("media")) -> Optional[Path]:
    """
    Выбирает и валидирует случайный медиа-файл. При необходимости уменьшает изображение.
    """
    file = pick_random_media_file(media_dir)
    if not file:
        logger.warning("Не найден ни один подходящий медиа-файл для публикации.")
        return None
    result = validate_media_file(file, media_dir)
    if not result.valid:
        logger.error(f"Медиа-файл не прошёл валидацию: {file}. Причина: {result.status} ({result.message})")
        return None
    media_type = get_media_type(file)
    if media_type == "image":
        try:
            with Image.open(file) as img:
                if img.size[0] > MAX_IMAGE_SIZE[0] or img.size[1] > MAX_IMAGE_SIZE[1]:
                    resized = process_image(file)
                    if resized is not None:
                        return resized
                return file
        except Exception as e:
            logger.error(f"Ошибка открытия изображения: {e}")
            return None
    return file
