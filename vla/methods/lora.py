from lerobot.policies import PreTrainedPolicy

def apply_lora(policy: PreTrainedPolicy, rank: int = 32) -> PreTrainedPolicy:
    return policy.wrap_with_peft(
        peft_cli_overrides={"method_type": "LORA", "r": rank, "lora_alpha": rank}
    )
