import json
from pathlib import Path
from typing import Any

from vla.diagnostics import plain, write_run


class TrainingObserver:
    def __init__(self, name: str, config: dict[str, Any], device: str):
        self.name = name
        self.config = plain(config)
        self.path = Path("runs") / name / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.clearml = None
        write_run(name, {"status": "running", **self.config, "metrics": str(self.path)})

        if device == "cuda":
            from clearml import Task

            self.clearml = Task.current_task() or Task.init(project_name="VLA cost curve", task_name=name)
            self.clearml.connect(plain(self.config), name=f"runs/{name}")

    def log(self, step: int, values: dict[str, Any]) -> None:
        row = {"step": step, **plain(values)}
        with self.path.open("a") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.clearml:
            logger = self.clearml.get_logger()
            for key, value in values.items():
                if isinstance(value, int | float):
                    logger.report_scalar("train", key, float(value), iteration=step)

    def finish(self, checkpoint: Path) -> None:
        write_run(
            self.name,
            {
                "status": "completed",
                **self.config,
                "metrics": str(self.path),
                "checkpoint": str(checkpoint),
            },
        )
        if self.clearml:
            self.clearml.upload_artifact("metrics", artifact_object=str(self.path))
            self.clearml.flush()

    def fail(self, error: Exception) -> None:
        write_run(
            self.name,
            {
                "status": "failed",
                **self.config,
                "metrics": str(self.path),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        if self.clearml:
            self.clearml.flush()


def log_evaluation(name: str, row: dict[str, Any], device: str) -> None:
    if device != "cuda":
        return
    from clearml import Task

    task = Task.current_task() or Task.init(project_name="VLA cost curve", task_name=name)
    task.connect(plain(row), name=f"evaluations/{name}")
    task.get_logger().report_scalar("eval", "success", row["success"], iteration=row["n_demos"])
    task.flush()
