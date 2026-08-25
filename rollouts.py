"""Rollout videos -> the frames that show where the trajectory breaks.

lerobot already writes a video per episode, but forty of them are not a diagnosis.
Task 3 asks for three characteristic failures with frames, so what is needed is a
function that finds the failed episodes and pulls a few frames out of each; the
report and the contact sheet both use it, and neither re-implements it.
"""

import base64
import io
import json
import pathlib

import imageio.v3 as iio
import numpy as np
from PIL import Image

def episodes(logs="eval_logs"):
    """-> [(run, task_id, episode, success, video path)] по всем оценкам на диске."""
    out = []
    for info_path in sorted(pathlib.Path(logs).glob("*/eval_info.json")):
        info = json.loads(info_path.read_text())
        for task in info.get("per_task", []):
            m = task["metrics"]
            for i, (ok, vid) in enumerate(zip(m["successes"], m["video_paths"])):
                path = pathlib.Path(vid)
                if not path.exists():
                    path = info_path.parent / path.relative_to(path.parts[0])
                if path.exists():
                    out.append((info_path.parent.name, task["task_id"], i, bool(ok), path))
    return out

def frames(path, n=4):
    """n кадров, равномерно по эпизоду, плюс общее число кадров."""
    video = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(video) - 1, n).astype(int)
    return [video[i] for i in idx], len(video)

def as_data_uri(frame, width=192, quality=70):
    """Кадр -> data:-строка, чтобы страница осталась самодостаточной."""
    image = Image.fromarray(np.asarray(frame))
    image = image.resize((width, round(width * image.height / image.width)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
