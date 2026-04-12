import os
from pathlib import Path

from skill.config import load_dotenv_file


def test_load_dotenv_file_sets_missing_values_only(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "TAVILY_API_KEY='from-file'",
                "EXTRA = spaced",
                "EMPTY=",
                "INVALID LINE",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("EXTRA", "from-env")
    monkeypatch.delenv("EMPTY", raising=False)

    loaded = load_dotenv_file(env_path)

    assert loaded is True
    assert os.environ["TAVILY_API_KEY"] == "from-file"
    assert os.environ["EXTRA"] == "from-env"
    assert os.environ["EMPTY"] == ""


def test_load_dotenv_file_returns_false_when_file_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / ".env"
    assert load_dotenv_file(missing_path) is False
