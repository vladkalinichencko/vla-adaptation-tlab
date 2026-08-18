#!/usr/bin/env bash
# Топорный перебор приёмов адаптации на 5 демо целевой задачи 0.
#
# Дефолт SmolVLA уже консервативный: freeze_vision_encoder=true, train_expert_only=true.
# Поэтому сетка такая: дефолт против полного размораживания против LoRA.
#
#   bash tmp/tricks_grid.sh
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=cgl PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
EP="399 405 410 421 437"      # первые 5 демо задачи 0 (task_index 19)
STEPS=1500
BS=2          # batch 4 роняет Metal, см. NOTES

run_one () {
  local tag="$1"; shift
  echo "=== $tag ==="
  if [ ! -d "outputs/$tag" ]; then
    $PY baseline.py --device mps train --tag "$tag" --episodes $EP \
      --steps $STEPS --batch-size $BS --seed 0 "$@" > "tmp/trick_${tag}_train.log" 2>&1
    echo "  обучение: exit=$?"
  fi
  # проверяем, что Metal ничего не отбросил, иначе числа мусорные
  errs=$(grep -ac "command buffer exited with error" "tmp/trick_${tag}_train.log" 2>/dev/null; true)
  errs=${errs:-0}
  if [ "$errs" -gt 0 ]; then
    echo "  !! $errs ошибок Metal — прогон недостоверен, пропускаю оценку"
    return
  fi
  $PY baseline.py --device mps eval \
    --policy "outputs/$tag/checkpoints/last/pretrained_model" \
    --task-id 0 --method "$tag" --n-demos 5 --seed 0 --tag "$tag" \
    > "tmp/trick_${tag}_eval.log" 2>&1
  echo "  оценка: exit=$? | $(grep -E '^success' tmp/trick_${tag}_eval.log || echo нет)"
}

run_one trick_default
run_one trick_unfrozen --policy.freeze_vision_encoder=false --policy.train_expert_only=false
run_one trick_lora     --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=32

echo
echo "=== итог ==="
$PY cost_curve.py runs/results.jsonl
