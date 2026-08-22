import json
import os
from pathlib import Path

os.environ["HF_HOME"] = str(Path("datasets/cache/huggingface").resolve())

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from convert_libero import demo_names, image_for_lerobot


SOURCE = Path(
    "/private/tmp/libero-inspect/libero_90/"
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5"
)
DATASET = Path("/private/tmp/official_libero_90_v3_test")
REPORT = Path("runs/diagnostics/conversion_test.json")
SHEET = Path("runs/diagnostics/conversion_test.png")


def decoded_image(frame: dict, key: str) -> np.ndarray:
    image = frame[key].permute(1, 2, 0).numpy()
    return np.rint(image.clip(0, 1) * 255).astype(np.uint8)


dataset = LeRobotDataset("local/official_libero_90_v3", root=DATASET, video_backend="pyav")
actual = (
    dataset.meta.total_episodes,
    dataset.meta.total_frames,
    dataset.meta.total_tasks,
    dataset.meta.fps,
)
assert actual == (50, 3762, 1, 20)

samples = []
tiles = []
with h5py.File(SOURCE) as source:
    data = source["data"]
    task = json.loads(data.attrs["problem_info"])["language_instruction"]
    names = demo_names(data)
    offsets = np.cumsum([0] + [len(data[name]["actions"]) for name in names])
    for episode in (0, len(names) // 2, len(names) - 1):
        demo = data[names[episode]]
        for local_frame in (0, len(demo["actions"]) // 2, len(demo["actions"]) - 1):
            output = dataset[int(offsets[episode] + local_frame)]
            state = np.concatenate(
                (demo["obs/ee_states"][local_frame], demo["obs/gripper_states"][local_frame])
            ).astype(np.float32)
            action = demo["actions"][local_frame].astype(np.float32)
            assert np.array_equal(output["observation.state"].numpy(), state)
            assert np.array_equal(output["action"].numpy(), action)
            assert output["episode_index"].item() == episode
            assert output["frame_index"].item() == local_frame
            assert output["task"] == task

            row = {"episode": episode, "frame": local_frame}
            images = []
            for raw_key, output_key, label in (
                ("obs/agentview_rgb", "observation.images.image", "front"),
                ("obs/eye_in_hand_rgb", "observation.images.image2", "wrist"),
            ):
                original = image_for_lerobot(demo[raw_key][local_frame])
                decoded = decoded_image(output, output_key)
                difference = np.abs(original.astype(np.int16) - decoded.astype(np.int16)).astype(
                    np.uint8
                )
                row[f"{label}_pixel_mae"] = float(difference.mean())
                magnified = np.minimum(difference.astype(np.uint16) * 4, 255).astype(np.uint8)
                images.extend((original, decoded, magnified))
            samples.append(row)
            tiles.append(images)

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(
    json.dumps({"episodes": 50, "frames": 3762, "fps": 20, "samples": samples}, indent=2)
    + "\n"
)
sheet = Image.new("RGB", (6 * 256, len(tiles) * 256))
for row, images in enumerate(tiles):
    for column, image in enumerate(images):
        sheet.paste(Image.fromarray(image), (column * 256, row * 256))
sheet.save(SHEET)
print(REPORT)
print(SHEET)
