from pathlib import Path
from typing import Optional, Set

def validate_path(
    path: Path,
    allowed_dir: Path,
    allowed_exts: Optional[Set[str]] = None,
    max_size_mb: int = 100,
    check_symlink: bool = True
) -> (bool, str):
    try:
        path = path.resolve(strict=True)
        allowed_dir = allowed_dir.resolve(strict=True)
        if check_symlink and path.is_symlink():
            return False, "Файл является symlink"
        if not (allowed_dir in path.parents or path == allowed_dir):
            return False, "Файл вне разрешённой директории"
        if allowed_exts and path.suffix.lower() not in allowed_exts:
            return False, f"Недопустимое расширение: {path.suffix}"
        if path.stat().st_size > max_size_mb * 1024 * 1024:
            return False, f"Файл слишком большой (> {max_size_mb} МБ)"
        return True, "OK"
    except Exception as e:
        return False, f"Ошибка валидации пути: {e}"
