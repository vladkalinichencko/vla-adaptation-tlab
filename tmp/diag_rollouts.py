"""Разбор роллаутов: где именно траектория ломается.

Видео lerobot уже пишет, но смотреть 40 штук глазами бессмысленно. Здесь берём
кадры из провальных эпизодов в характерных точках и складываем в контактный лист,
чтобы увидеть общий паттерн: робот не доезжает, промахивается схватом, или
хватает и роняет.

    python tmp/diag_rollouts.py --logs eval_logs --out tmp/rollout_frames.png
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_frames(path, n=4):
    """n кадров, равномерно по эпизоду."""
    import imageio.v3 as iio
    frames = iio.imread(path, plugin="pyav")
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    return [frames[i] for i in idx], len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="eval_logs")
    p.add_argument("--out", default="tmp/rollout_frames.png")
    p.add_argument("--episodes", type=int, default=4, help="сколько эпизодов показать")
    p.add_argument("--frames", type=int, default=4)
    args = p.parse_args()

    runs = sorted(pathlib.Path(args.logs).glob("*/eval_info.json"))
    if not runs:
        raise SystemExit(f"нет eval_info.json в {args.logs}")

    picked = []
    for info_path in runs:
        info = json.loads(info_path.read_text())
        for task in info.get("per_task", []):
            m = task["metrics"]
            for i, (ok, vid) in enumerate(zip(m["successes"], m["video_paths"])):
                vp = pathlib.Path(vid)
                if not vp.exists():
                    vp = info_path.parent / pathlib.Path(vid).relative_to(
                        pathlib.Path(vid).parts[0])
                if vp.exists():
                    picked.append((info_path.parent.name, task["task_id"], i, ok, vp))
    fails = [x for x in picked if not x[3]][: args.episodes]
    wins = [x for x in picked if x[3]][: max(0, args.episodes - len(fails))]
    show = fails + wins
    if not show:
        raise SystemExit("видео не найдены")

    print(f"эпизодов всего {len(picked)}, провалов {sum(1 for x in picked if not x[3])}")
    fig, axes = plt.subplots(len(show), args.frames,
                             figsize=(3 * args.frames, 2.6 * len(show)), squeeze=False)
    for row, (run, task, ep, ok, vp) in enumerate(show):
        frames, total = read_frames(vp, args.frames)
        for col, fr in enumerate(frames):
            axes[row][col].imshow(fr)
            axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
            if col == 0:
                axes[row][col].set_ylabel(f"{run}\nэп.{ep} {'успех' if ok else 'ПРОВАЛ'}",
                                          fontsize=7)
            axes[row][col].set_title(f"шаг {int(col * (total - 1) / (args.frames - 1))}",
                                     fontsize=8)
        print(f"  {run} задача {task} эпизод {ep}: {'успех' if ok else 'провал'}, "
              f"{total} кадров")
    fig.suptitle("кадры роллаутов: строка — эпизод, столбцы — по ходу эпизода")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
