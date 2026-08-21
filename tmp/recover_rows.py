"""Восстановить строки results.jsonl из уже посчитанных eval_info.json.

Оценка отработала, а запись падала на разборе — пересчитывать 20 эпизодов заново
незачем, все числа уже лежат на диске.
"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import baseline

rows = []
for info_path in sorted(pathlib.Path("eval_logs").glob("*/eval_info.json")):
    tag = info_path.parent.name
    m = re.search(r"_t(\d+)$", tag)
    if not m:
        print(f"пропускаю {tag}: не понял task_id")
        continue
    agg = baseline.aggregated(json.loads(info_path.read_text()))
    rows.append({
        "method": tag.rsplit("_t", 1)[0],
        "seed": 0,
        "task": f"libero_goal_{m.group(1)}",
        "n_demos": 0 if "zeroshot" in tag else 5,
        "success": agg["pc_success"] / 100,
        "n_episodes": agg["n_episodes"],
        "policy": tag,
    })

baseline.RESULTS.parent.mkdir(exist_ok=True)
seen = set()
if baseline.RESULTS.exists():
    for line in baseline.RESULTS.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            seen.add((d["method"], d["task"], d["seed"], d["n_demos"]))

with baseline.RESULTS.open("a") as f:
    for r in rows:
        if (r["method"], r["task"], r["seed"], r["n_demos"]) in seen:
            continue
        f.write(json.dumps(r) + "\n")
        print(f"{r['task']:<16} {r['method']:<16} success={r['success']:.2f} "
              f"({r['n_episodes']} эп.)")
