#!/usr/bin/env bash
# Кривая по бюджетам демо для одной задачи и одного сида + добор LoRA.
# Полная кривая (3 задачи x 2 сида) — это сутки, здесь один срез для проверки формы.
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=cgl PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
TASK_ID=0
TASK_INDEX=19
STEPS=1500
BS=2

train_eval () {
  local tag="$1" n="$2"; shift 2
  local eps
  eps=$($PY demo_subset.py --task-index $TASK_INDEX --n "$n" | grep -oE '\[[0-9,]+\]$' | tr -d '[]' | tr ',' ' ')
  echo "=== $tag ($n демо: $eps) ==="
  if [ ! -d "outputs/$tag" ]; then
    $PY baseline.py --device mps train --tag "$tag" --episodes $eps \
      --steps $STEPS --batch-size $BS --seed 0 "$@" > "tmp/${tag}_train.log" 2>&1
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

train_eval trick_lora 5 --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=32
train_eval default_n10 10
train_eval default_n25 25

echo
echo "=== кривая ==="
$PY cost_curve.py runs/results.jsonl
