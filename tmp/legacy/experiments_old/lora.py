from lerobot.configs.default import PeftConfig
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from vla.data import SEEN_CHECKPOINT, TARGET_SOURCE, dataset, first_target_episodes
from vla.evaluation import evaluate
from vla.runtime import adaptation_cells, current_runtime, training_steps
from vla.training import load_policy, train_policy


def run():
    runtime = current_runtime()
    rows = []
    for seed, task_id, demos in adaptation_cells(runtime):
        name = f"lora_t{task_id}_n{demos}_s{seed}"
        steps = training_steps(runtime, demos)
        policy = load_policy(SEEN_CHECKPOINT, None, runtime)
        data = dataset(TARGET_SOURCE, first_target_episodes(task_id, demos))
        checkpoint = train_policy(
            name,
            policy,
            data,
            steps,
            seed,
            runtime,
            peft=PeftConfig(method_type="LORA", r=32, lora_alpha=32),
            optimizer=AdamWConfig(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10),
            scheduler=CosineDecayWithWarmupSchedulerConfig(
                peak_lr=1e-3,
                decay_lr=1e-5,
                num_warmup_steps=100,
                num_decay_steps=steps,
            ),
        )
        rows.append(evaluate(name, checkpoint, "lora_r32", task_id, demos, seed, runtime))
    return rows


if __name__ == "__main__":
    run()
