#!/usr/bin/env bash
# Приёмы адаптации на 5 демо целевой задачи 0, по одному, при равном числе шагов.
#
# Дефолт SmolVLA уже консервативный: freeze_vision_encoder=true, train_expert_only=true.
# Каждый приём — одна гипотеза; предсказание пишется в NOTES до запуска.
#
#   ./run_tricks.sh            # все приёмы
#   ./run_tricks.sh mix chunk  # только названные
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=cgl PYTORCH_ENABLE_MPS_FALLBACK=1
PY=".venv/bin/python"
EP="399 405 410 421 437"      # первые 5 демо задачи 0 (task_index 19)
# по 4 демо каждой из десяти задач libero_object: единственные seen-задачи в
# lerobot/libero — libero_90 в этот датасет не входит, см. NOTES
SEEN="807 829 832 835 808 809 815 820 810 856 862 886 811 812 824 843 813 818 825 850 \
814 817 842 851 816 822 831 865 819 827 828 844 821 830 839 864 823 826 834 840"
STEPS=1500
BS=2          # batch 4 роняет Metal, см. NOTES

run_one () {
  local tag="$1"; local eps="$2"; shift 2
  echo "=== $tag ==="
  if [ ! -d "outputs/$tag" ]; then
    $PY baseline.py --device mps train --tag "$tag" --episodes $eps \
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

SELECT=("$@")
[ ${#SELECT[@]} -eq 0 ] && SELECT=(default unfrozen lora mix chunk aug)
has () { for w in "${SELECT[@]}"; do [ "$w" = "$1" ] && return 0; done; return 1; }

has default  && run_one trick_default  "$EP"
has unfrozen && run_one trick_unfrozen "$EP" --policy.freeze_vision_encoder=false --policy.train_expert_only=false
has lora     && run_one trick_lora     "$EP" --peft.method_type=LORA --peft.r=32 --peft.lora_alpha=32
has mix      && run_one trick_mix      "$EP $SEEN"
has chunk    && run_one trick_chunk    "$EP" --policy.chunk_size=10 --policy.n_action_steps=10
has aug      && run_one trick_aug      "$EP" --dataset.image_transforms.enable=true

echo
echo "=== итог ==="
$PY cost_curve.py runs/results.jsonl
