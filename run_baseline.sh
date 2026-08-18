#!/usr/bin/env bash
# Задача 1 целиком: zero-shot, контроль на язык, наивный файнтюн 5/10/25, cost curve.
# Запускать на Linux с GPU после setup_gpu.sh. Всё пишется в runs/results.jsonl.
#
#   ./run_baseline.sh                 # весь бейзлайн
#   DRY=1 ./run_baseline.sh           # только показать команды
#
# Перед первым запуском сверьте соответствие task_id сьюта и task_index датасета:
#   python demo_subset.py --suite libero_goal
# и впишите TASK_INDICES ниже (порядок должен совпадать с TASK_IDS).

set -euo pipefail
cd "$(dirname "$0")"

# работаем из venv проекта, даже если он не активирован
PY="python"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

SEEN_CKPT="${SEEN_CKPT:-lerobot/smolvla_base}"   # свой seen-чекпойнт после претрена на libero_90
DATASET="${DATASET:-lerobot/libero}"
TASK_IDS=(0 1 2)                                  # --env.task_ids, порядок сьюта libero_goal
TASK_INDICES=(19 17 14)                           # те же задачи в датасете (см. --suite libero_goal)
BUDGETS=(5 10 25)
SEEDS=(0 1)
EPISODES="${EPISODES:-20}"
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
STEPS="${STEPS:-20000}"
DRY_FLAG=""
[ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

echo "### Точка 0: zero-shot seen-чекпойнта"
for i in "${!TASK_IDS[@]}"; do
  "$PY" baseline.py $DRY_FLAG eval \
    --policy "$SEEN_CKPT" --task-id "${TASK_IDS[$i]}" \
    --method zeroshot --n-demos 0 --seed 0 --tag "zeroshot_t${TASK_IDS[$i]}"
done

echo
echo "### Бейзлайн: наивный файнтюн на 5/10/25 демо"
for seed in "${SEEDS[@]}"; do
  for i in "${!TASK_IDS[@]}"; do
    task_id="${TASK_IDS[$i]}"
    task_index="${TASK_INDICES[$i]}"
    for n in "${BUDGETS[@]}"; do
      tag="base_t${task_id}_n${n}_s${seed}"
      episodes=$("$PY" demo_subset.py --repo-id "$DATASET" --task-index "$task_index" --n "$n" \
                 | grep -oE '\[[0-9,]+\]$' | tr -d '[]')
      echo "--- $tag  эпизоды: $episodes"

      "$PY" baseline.py $DRY_FLAG train \
        --ckpt "$SEEN_CKPT" --dataset "$DATASET" \
        --episodes ${episodes//,/ } --steps "$STEPS" --seed "$seed" --tag "$tag"

      "$PY" baseline.py $DRY_FLAG eval \
        --policy "outputs/$tag/checkpoints/last/pretrained_model" \
        --task-id "$task_id" --method baseline --n-demos "$n" --seed "$seed" --tag "$tag"
    done
  done
done

echo
echo "### Cost curve"
"$PY" cost_curve.py runs/results.jsonl --plot
