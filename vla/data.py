import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow
from lerobot.configs.default import DatasetConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

BASE_POLICY = "lerobot/smolvla_base"
BASE_POLICY_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
SEEN = "local/official_libero_90_v3"
SEEN_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SEEN_ROOT = Path("datasets/lerobot/official_libero_90_v3")
TARGET = "local/official_libero_goal_v3"
TARGET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
TARGET_ROOT = Path("datasets/lerobot/official_libero_goal_v3")
TARGET_SUITE = "libero_goal"
TARGET_INSTRUCTIONS = (
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
)
SEEN_CHECKPOINT = Path("outputs/seen_pretrain/checkpoints/last/pretrained_model")
RENAME = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
CACHE = Path("datasets")

@dataclass(frozen=True)
class Source:
    repo_id: str
    revision: str
    root: Path | None = None

SEEN_SOURCE = Source(SEEN, SEEN_REVISION, SEEN_ROOT)
TARGET_SOURCE = Source(TARGET, TARGET_REVISION, TARGET_ROOT)

@dataclass(frozen=True)
class Mix:
    source: Source
    target_frames: int
    total_frames: int

def metadata(source: Source) -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata(source.repo_id, root=source.root, revision=source.revision)

def episode_tasks(source: Source) -> dict[int, int]:
    cache = CACHE / f"episode_tasks_{source.repo_id.replace('/', '_')}_{source.revision[:8]}.json"
    if cache.exists():
        return {int(episode): int(task) for episode, task in json.loads(cache.read_text()).items()}

    data = str(source.root / "data") if source.root else f"hf://datasets/{source.repo_id}@{source.revision}/data"
    table = arrow.dataset(data, format="parquet").to_table(columns=["episode_index", "task_index"])
    mapping: dict[int, int] = {}
    for episode, task in zip(table["episode_index"].to_pylist(), table["task_index"].to_pylist()):
        mapping.setdefault(int(episode), int(task))
    cache.write_text(json.dumps(mapping, sort_keys=True) + "\n")
    return mapping

def first_target_episodes(task_id: int, count: int) -> list[int]:
    task_index = metadata(TARGET_SOURCE).get_task_index(TARGET_INSTRUCTIONS[task_id])
    if task_index is None:
        raise ValueError(f"Target instruction is absent: {TARGET_INSTRUCTIONS[task_id]}")
    episodes = [episode for episode, task in sorted(episode_tasks(TARGET_SOURCE).items()) if task == task_index]
    if len(episodes) < count:
        raise ValueError(f"Task {task_id} has {len(episodes)} episodes, requested {count}.")
    return episodes[:count]

def balanced_seen_episodes(count: int) -> list[int]:
    by_task: dict[int, list[int]] = {}
    for episode, task in sorted(episode_tasks(SEEN_SOURCE).items()):
        by_task.setdefault(task, []).append(episode)
    selected: list[int] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for task in sorted(by_task):
            if round_index < len(by_task[task]):
                selected.append(by_task[task][round_index])
                added = True
                if len(selected) == count:
                    return selected
        if not added:
            raise ValueError(f"Seen dataset has fewer than {count} episodes.")
        round_index += 1
    return selected

def dataset(source: Source, episodes: list[int] | None = None) -> DatasetConfig:
    return DatasetConfig(
        repo_id=source.repo_id,
        root=str(source.root) if source.root else None,
        revision=source.revision,
        episodes=episodes,
    )

MIX_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
}

def _copy_episodes(output: LeRobotDataset, source: Source, episodes: list[int], stride: int) -> int:
    data = LeRobotDataset(
        source.repo_id,
        root=source.root,
        revision=source.revision,
        episodes=episodes,
        video_backend="pyav",
    )
    current_episode = None
    frames = 0
    for index in range(len(data)):
        frame = data[index]
        episode = int(frame["episode_index"])
        if current_episode is not None and episode != current_episode:
            output.save_episode(parallel_encoding=False)
        current_episode = episode
        if int(frame["frame_index"]) % stride:
            continue
        output.add_frame(
            {
                "observation.images.image": _image(frame["observation.images.image"]),
                "observation.images.image2": _image(frame["observation.images.image2"]),
                "observation.state": frame["observation.state"].numpy().astype(np.float32),
                "action": frame["action"].numpy().astype(np.float32),
                "task": frame["task"],
            }
        )
        frames += 1
    if current_episode is not None:
        output.save_episode(parallel_encoding=False)
    return frames

def _image(image) -> np.ndarray:
    return np.rint(image.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

def build_mix(task_id: int, demos: int) -> Mix:
    root = Path("datasets/mixes") / f"goal_{task_id}_n{demos}"
    manifest = root / "mix.json"
    if root.exists():
        if not manifest.is_file():
            raise FileExistsError(f"Mixed dataset is incomplete: {root}")
        saved = json.loads(manifest.read_text())
        return Mix(Source(saved["repo_id"], "local", root), saved["target_frames"], saved["total_frames"])
    output = LeRobotDataset.create(
        repo_id=f"local/libero_mix_goal_{task_id}_n{demos}",
        root=root,
        robot_type="panda",
        fps=20,
        features=MIX_FEATURES,
        use_videos=True,
        image_writer_threads=4,
    )
    target_frames = _copy_episodes(output, TARGET_SOURCE, first_target_episodes(task_id, demos), 1)
    seen_frames = _copy_episodes(output, SEEN_SOURCE, balanced_seen_episodes(demos), 1)
    output.finalize()
    source = Source(output.repo_id, "local", root)
    mix = Mix(source, target_frames, target_frames + seen_frames)
    manifest.write_text(
        json.dumps(
            {
                "repo_id": source.repo_id,
                "fps": 20,
                "target_fps": 20,
                "seen_source_fps": 20,
                "seen_stride": 1,
                "target_frames": mix.target_frames,
                "total_frames": mix.total_frames,
            },
            indent=2,
        )
        + "\n"
    )
    return mix
