from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


def make_latent_smolvla_pre_post_processors(config, dataset_stats=None):
    return make_smolvla_pre_post_processors(config, dataset_stats)
