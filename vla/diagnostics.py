import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


RUNS = Path("runs")


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value


def write_run(name: str, row: dict[str, Any]) -> None:
    path = RUNS / name / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(row), indent=2, ensure_ascii=False) + "\n")


def append_result(row: dict[str, Any]) -> None:
    RUNS.mkdir(exist_ok=True)
    with (RUNS / "results.jsonl").open("a") as file:
        file.write(json.dumps(plain(row), ensure_ascii=False) + "\n")


def require_checkpoint(output: Path) -> Path:
    checkpoint = output / "checkpoints" / "last" / "pretrained_model"
    config = checkpoint / "config.json"
    if not checkpoint.is_dir() or not config.is_file():
        raise RuntimeError(f"Training finished without its final checkpoint: {output}")
    return checkpoint
