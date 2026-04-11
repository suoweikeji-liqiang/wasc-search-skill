from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist() -> None:
    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "SETUP.md").exists()
    assert (REPO_ROOT / "skill/SKILL.md").exists()
    assert (REPO_ROOT / "scripts/run_shadow_benchmark.py").exists()


def test_required_files_exist_outside_repo_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    assert (REPO_ROOT / "README.md").exists()
    assert (REPO_ROOT / "SETUP.md").exists()
    assert (REPO_ROOT / "skill/SKILL.md").exists()
    assert (REPO_ROOT / "scripts/run_shadow_benchmark.py").exists()
