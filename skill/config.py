from pathlib import Path
import os


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    normalized_key = key.strip()
    if not normalized_key:
        return None

    normalized_value = value.strip().strip("'\"")
    return normalized_key, normalized_value


def load_dotenv_file(path: str | Path = ".env") -> bool:
    env_path = Path(path)
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
    return True
