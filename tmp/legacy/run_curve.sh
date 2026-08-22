#!/usr/bin/env bash
# Кривая по бюджетам демо для одной задачи и одного сида + добор LoRA.
# Полная кривая (3 задачи x 2 сида) — это сутки, здесь один срез для проверки формы.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=cgl PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
TASK_ID=0
TASK_INDEX=19
# число шагов масштабируем по числу демо: иначе при 25 демо модель проходит по
# данным меньше одного раза, а при 5 — четыре, и кривая падает из-за недообучения
STEPS_PER_DEMO=300
BS=2

train_eval () {
  local tag="$1" n="$2"; shift 2
  local eps
  eps=$($PY demo_subset.py --task-index $TASK_INDEX --n "$n" | grep -oE '\[[0-9,]+\]$' | tr -d '[]' | tr ',' ' ')
  echo "=== $tag ($n демо: $eps) ==="
  local steps=$(( n * STEPS_PER_DEMO ))
  if [ ! -d "outputs/$tag" ]; then
    echo "  шагов: $steps (по $STEPS_PER_DEMO на демо)"
    $PY baseline.py --device mps train --tag "$tag" --episodes $eps \
      --steps $steps --batch-size $BS --seed 0 "$@" > "tmp/${tag}_train.log" 2>&1
    echo "  обучение exit=$?"
  fi
  local errs
  errs=$(grep -ac "command buffer exited with error" "tmp/${tag}_train.log" 2>/dev/null; true)
  if [ "${errs:-0}" -gt 0 ]; then
    echo "  !! ${errs} ошибок Metal — прогон недостоверен, оценку пропускаю"
    return
  fi
  $PY baseline.py --device mps eval \
    --policy "outputs/$tag/checkpoints/last/pretrained_model" \
    --task-id $TASK_ID --method "${tag%%_n*}" --n-demos "$n" --seed 0 --tag "$tag" \
    > "tmp/${tag}_eval.log" 2>&1
  echo "  оценка exit=$? | $(grep -aE '^success' tmp/${tag}_eval.log || echo нет)"
}

train_eval fixed_n5  5
train_eval fixed_n10 10
train_eval fixed_n25 25

echo
echo "=== кривая ==="
$PY cost_curve.py runs/results.jsonl
