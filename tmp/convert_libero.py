import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("datasets/hf_cache").resolve()))

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"

FEATURES = {
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


def image_for_lerobot(image: np.ndarray) -> np.ndarray:
    rotated = np.flip(image, axis=(0, 1))
    return np.asarray(Image.fromarray(rotated).resize((256, 256), Image.Resampling.BICUBIC))


def demo_names(data: h5py.Group) -> list[str]:
    return sorted(data, key=lambda name: int(name.removeprefix("demo_")))


def add_file(dataset: LeRobotDataset, path: Path) -> tuple[int, int]:
    episodes = frames = 0
    with h5py.File(path) as source:
        data = source["data"]
        task = json.loads(data.attrs["problem_info"])["language_instruction"]
        for name in demo_names(data):
            demo = data[name]
            obs = demo["obs"]
            arrays = [
                demo["actions"],
                obs["agentview_rgb"],
                obs["eye_in_hand_rgb"],
                obs["ee_states"],
                obs["gripper_states"],
            ]
            lengths = {len(array) for array in arrays}
            if len(lengths) != 1:
                raise ValueError(f"Mismatched arrays in {path.name}/{name}: {sorted(lengths)}")
            numeric = (demo["actions"], obs["ee_states"], obs["gripper_states"])
            if not all(np.isfinite(array[:]).all() for array in numeric):
                raise ValueError(f"Non-finite state or action in {path.name}/{name}")

            for action, front, wrist, ee, gripper in zip(*arrays, strict=True):
                dataset.add_frame(
                    {
                        "observation.images.image": image_for_lerobot(front),
                        "observation.images.image2": image_for_lerobot(wrist),
                        "observation.state": np.concatenate((ee, gripper)).astype(np.float32),
                        "action": action.astype(np.float32),
                        "task": task,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            episodes += 1
            frames += lengths.pop()
    return episodes, frames


def convert(files: list[Path], output: Path, repo_id: str) -> dict:
    if output.exists():
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=output,
            image_writer_threads=8,
        )
        if dataset.meta.total_episodes % 50:
            raise ValueError("Existing conversion stops inside a source file")
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output,
            robot_type="panda",
            fps=20,
            features=FEATURES,
            use_videos=True,
            image_writer_threads=8,
        )

    completed_files = dataset.meta.total_episodes // 50
    totals = {
        "files": len(files),
        "episodes": dataset.meta.total_episodes,
        "frames": dataset.meta.total_frames,
    }
    for index, path in enumerate(files[completed_files:], completed_files + 1):
        episodes, frames = add_file(dataset, path)
        totals["episodes"] += episodes
        totals["frames"] += frames
        print(f"{index}/{len(files)} {path.name}: {episodes} episodes, {frames} frames", flush=True)
        dataset.finalize()
        if index < len(files):
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=output,
                image_writer_threads=8,
            )
    return totals


def convert_suite(suite: str, expected_files: int) -> dict:
    source = Path("datasets/raw/official_libero") / suite
    output = Path("datasets/lerobot") / f"official_{suite}_v3"
    repo_id = f"local/official_{suite}_v3"
    files = sorted(source.glob("*.hdf5"))
    if len(files) != expected_files:
        raise ValueError(f"Expected {expected_files} source files, found {len(files)}")
    totals = convert(files, output, repo_id)
    expected_episodes = expected_files * 50
    if totals["episodes"] != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {totals['episodes']}")
    (Path("logs") / f"{suite}_conversion.json").write_text(
        json.dumps(
            {
                "source_revision": REVISION,
                "fps": 20,
                "images": "rotate 180 degrees, bicubic 128x128 -> 256x256",
                **totals,
            },
            indent=2,
        )
        + "\n"
    )
    print(totals)
    return totals


if __name__ == "__main__":
    convert_suite("libero_goal", 10)
