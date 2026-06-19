"""rlhf — a from-scratch RLHF pipeline.

Components:
  - data/        preference + prompt + SFT datasets
  - models/      reward model (scalar head) and actor-critic policy (value head)
  - algorithms/  reward-model trainer, PPO (from scratch), DPO, GRPO
  - utils/       config, logging, generation, and RL/tensor math

The reinforcement-learning and reward-modelling logic is implemented by hand;
HuggingFace `transformers` is used only for the pretrained backbones + tokenizers.
"""

__version__ = "0.1.0"
