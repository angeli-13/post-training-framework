#!/usr/bin/env python3
"""
Axolotl/TRL config generators with shared, composable defaults.

Design
------
A small OO hierarchy maximizes reuse while keeping comments and behavior self-contained:

Classes
  BaseConfigBuilder   : Builds algorithm-agnostic fields (model, tokenizer/chat, IO, dtype, W&B, resume, adapters).
  SFTConfigBuilder    : Adds SFT-specific dataset block.
  DPOConfigBuilder    : Adds preference-training dataset block and TRL params for DPO.
  GRPOConfigBuilder   : Adds RL-specific dataset transform and TRL (GRPO) block.
  PPOConfigBuilder    : Adds PPO dataset transform (query output), PPO hyperparams, rollout/generation, and rewards.

Why a single file right now?
- The classes share constants (MODEL_DEFAULTS) and helpers (model-key resolution, YAML writer).
- Co-locating avoids duplication and keeps field ordering predictable. If this grows, it can be refactored
  into a small package (builders/base.py, sft.py, dpo.py, grpo.py, ppo.py, common.py) without changing the public CLI.

Usage examples
--------------
SFT:
  python config_builders.py sft \
    --base_model /work/.../Qwen3-4B-Thinking-2507 \
    --dataset_path /data/chunk1-merged.jsonl \
    --output_path qwen3_thinking_sft.yml \
    --adapter full \
    --output_dir outputs/qwen3-sft \
    --sequence_len 2048 \
    --micro_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --wandb_project hhri-foxbrain \
    --wandb_name qwen3-sft-run1

DPO (Argilla-style pairwise):
  python config_builders.py dpo \
    --base_model /work/.../Qwen3-4B-Instruct-2507 \
    --dataset_path /data/dpo_pairs.jsonl \
    --output_path qwen3_dpo.yml \
    --output_dir outputs/qwen3-dpo \
    --sequence_len 4096 \
    --micro_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --trl_beta 0.1 \
    --dpo_label_smoothing 0.0

GRPO:
  python config_builders.py grpo \
    --base_model /work/.../Qwen3-4B-Instruct-2507 \
    --dataset_path /data/grpo.jsonl \
    --output_path qwen3_grpo.yml \
    --output_dir outputs/qwen3-grpo \
    --sequence_len 4096 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --trl_num_generations 8 \
    --trl_max_completion_length 256 \
    --trl_reward_funcs rewards.model_helpfulness_reward,rewards.think_format_reward \
    --trl_reward_weights 1.0,0.2

PPO (YAML consumed by a standalone TRL runner):
  python config_builders.py ppo \
    --base_model /work/.../Qwen3-4B-Instruct-2507 \
    --dataset_path /data/ppo.jsonl \
    --output_path qwen3_ppo.yml \
    --output_dir outputs/qwen3-ppo \
    --sequence_len 4096 \
    --micro_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --ppo_batch_size 64 \
    --ppo_mini_batch_size 16 \
    --ppo_epochs 4 \
    --ppo_learning_rate 5e-6 \
    --ppo_target_kl 0.1 \
    --gen_max_new_tokens 256 \
    --gen_temperature 0.7 \
    --gen_top_p 0.9 \
    --reward_funcs rewards.model_helpfulness_reward,rewards.think_format_reward \
    --reward_weights 1.0,0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Model defaults and helpers
# ---------------------------------------------------------------------------
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Instruct variant — canonical tokenizer pad token
    "Qwen/Qwen3-4B-Instruct-2507": {
        "chat_template": "tokenizer_default",
        "special_tokens": {"pad_token": "<|endoftext|>"},
        "trust_remote_code": True,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    # Thinking variant — canonical tokenizer pad token
    "Qwen/Qwen3-4B-Thinking-2507": {
        "chat_template": "tokenizer_default",
        "special_tokens": {"pad_token": "<|endoftext|>"},
        "trust_remote_code": True,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
}


def resolve_model_key(base_model: str, explicit: Optional[str]) -> Optional[str]:
    """Return a known model key for defaults.

    Tries, in order: explicit value, exact match, substring match on the last
    path component of known keys inside the provided base_model string.
    """
    if explicit:
        return explicit
    if base_model in MODEL_DEFAULTS:
        return base_model
    lowered = base_model.lower()
    for key in MODEL_DEFAULTS:
        name = key.split("/")[-1].lower()
        if name in lowered:
            return key
    return None


# ---------------------------------------------------------------------------
# Base builder
# ---------------------------------------------------------------------------
class BaseConfigBuilder:
    """Build algorithm-agnostic Axolotl config sections.

    Subclasses should call `build_base()` then mutate the returned dict.
    """

    def __init__(
        self,
        base_model: str,
        dataset_path: str,
        output_dir: str,
        sequence_len: int = 4096,
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        val_set_size: float = 0.02,
        run_name: Optional[str] = None,
        chat_template: Optional[str] = None,
        model_key: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        auto_resume_from_checkpoints: Optional[bool] = None,
        # W&B
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        wandb_mode: Optional[str] = None,
        wandb_watch: Optional[str] = None,
        wandb_log_model: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        # Adapters (shared across SFT and GRPO)
        adapter: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.sequence_len = sequence_len
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.val_set_size = val_set_size
        self.run_name = run_name
        self.chat_template = chat_template
        self.model_key = model_key
        self.resume_from_checkpoint = resume_from_checkpoint
        self.auto_resume_from_checkpoints = auto_resume_from_checkpoints
        # W&B
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_name = wandb_name
        self.wandb_run_id = wandb_run_id
        self.wandb_mode = wandb_mode
        self.wandb_watch = wandb_watch
        self.wandb_log_model = wandb_log_model
        self.wandb_tags = wandb_tags
        # Adapters
        self.adapter = (adapter or "full").lower()
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

    def build_base(self) -> Dict[str, Any]:
        mdef = MODEL_DEFAULTS.get(self.model_key or self.base_model, {})
        cfg: Dict[str, Any] = {
            "base_model": self.base_model,
            "trust_remote_code": bool(mdef.get("trust_remote_code", False)),
            "chat_template": self.chat_template or mdef.get("chat_template", "tokenizer_default"),
            "datasets": [  # subclasses override `type` and dataset specifics
                {
                    "path": self.dataset_path,
                    "type": "chat_template",
                }
            ],
            "dataset_prepared_path": "last_run_prepared",
            "val_set_size": self.val_set_size,
            "sequence_len": self.sequence_len,
            "sample_packing": True,
            "eval_sample_packing": False,
            "pad_to_sequence_len": True,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler": "cosine",
            "warmup_ratio": self.warmup_ratio,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "bf16": True,
            "output_dir": self.output_dir,
        }
        # Attach model-specific special tokens when known
        if "special_tokens" in mdef:
            cfg["special_tokens"] = mdef["special_tokens"]
        # Optional W&B
        if self.wandb_project:
            cfg["wandb_project"] = self.wandb_project
        if self.wandb_entity:
            cfg["wandb_entity"] = self.wandb_entity
        if self.wandb_name:
            cfg["wandb_name"] = self.wandb_name
        if self.wandb_run_id:
            cfg["wandb_run_id"] = self.wandb_run_id
        if self.wandb_mode:
            cfg["wandb_mode"] = self.wandb_mode
        if self.wandb_watch:
            cfg["wandb_watch"] = self.wandb_watch
        if self.wandb_log_model:
            cfg["wandb_log_model"] = self.wandb_log_model
        if self.wandb_tags:
            cfg["wandb_tags"] = list(self.wandb_tags)
        # Resume
        if self.resume_from_checkpoint:
            cfg["resume_from_checkpoint"] = str(self.resume_from_checkpoint)
        if self.auto_resume_from_checkpoints is not None:
            cfg["auto_resume_from_checkpoints"] = bool(self.auto_resume_from_checkpoints)
        # Adapters shared across algorithms
        self._apply_adapter(cfg, mdef)
        return cfg

    def _apply_adapter(self, cfg: Dict[str, Any], mdef: Dict[str, Any]) -> None:
        """Attach LoRA/QLoRA configuration if requested.

        This logic is shared by SFT and GRPO builders.
        """
        if self.adapter not in {"lora", "qlora"}:
            return
        cfg["adapter"] = self.adapter
        if self.adapter == "qlora":
            cfg.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "optimizer": "paged_adamw_8bit",
            })
        else:
            cfg.update({
                "load_in_8bit": True,
                "optimizer": "adamw_torch",
            })
        cfg.update({
            "lora_r": int(self.lora_r),
            "lora_alpha": int(self.lora_alpha),
            "lora_dropout": float(self.lora_dropout),
        })
        tmods = self.lora_target_modules or mdef.get("lora_target_modules")
        if tmods:
            cfg["lora_target_modules"] = list(tmods)

    @staticmethod
    def write_yaml(config: Dict[str, Any], output_path: str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# SFT builder
# ---------------------------------------------------------------------------
class SFTConfigBuilder(BaseConfigBuilder):
    """Add SFT-specific dataset fields."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self) -> Dict[str, Any]:
        cfg = self.build_base()
        cfg["datasets"][0].update({
            "type": "chat_template",
            "roles_to_train": ["assistant"],
            "train_on_eos": "turn",
        })
        return cfg


# ---------------------------------------------------------------------------
# GRPO builder
# ---------------------------------------------------------------------------
class GRPOConfigBuilder(BaseConfigBuilder):
    """Add RL-specific dataset transform and TRL settings for GRPO."""

    def __init__(self, *args,
                 trl_num_generations: int = 4,
                 trl_max_completion_length: int = 256,
                 trl_reward_funcs: Optional[List[str]] = None,
                 trl_reward_weights: Optional[List[float]] = None,
                 trl_use_vllm: Optional[bool] = None,
                 trl_loss_type: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.trl_num_generations = trl_num_generations
        self.trl_max_completion_length = trl_max_completion_length
        self.trl_reward_funcs = trl_reward_funcs or [
            "rewards.model_helpfulness_reward",
            "rewards.think_format_reward",
        ]
        self.trl_reward_weights = trl_reward_weights or [1.0, 0.2]
        self.trl_use_vllm = trl_use_vllm
        self.trl_loss_type = trl_loss_type  # e.g., "dr_grpo"

    def build(self) -> Dict[str, Any]:
        cfg = self.build_base()
        # RL dataset: use a transform that returns {"prompt": ...}
        cfg["datasets"][0].update({
            "type": "rewards.messages_to_prompt_transform",
        })
        # Top-level RL selector (required by Axolotl)
        cfg["rl"] = "grpo"
        # TRL block (no 'rl' key inside)
        trl: Dict[str, Any] = {
            "num_generations": int(self.trl_num_generations),
            "max_completion_length": int(self.trl_max_completion_length),
            "reward_funcs": list(self.trl_reward_funcs),
            "reward_weights": list(self.trl_reward_weights),
        }
        if self.trl_use_vllm is not None:
            trl["use_vllm"] = bool(self.trl_use_vllm)
        if self.trl_loss_type:
            trl["loss_type"] = self.trl_loss_type
        cfg["trl"] = trl
        return cfg


# ---------------------------------------------------------------------------
# DPO builder
# ---------------------------------------------------------------------------
class DPOConfigBuilder(BaseConfigBuilder):
    """Add preference-learning dataset mapping and TRL params for DPO.

    Expects a dataset row like:
      {"system": "...", "instruction": "...", "chosen_response": "...", "rejected_response": "..."}
    which corresponds to Axolotl's `chatml.argilla` DPO type.
    """

    def __init__(self, *args,
                 trl_beta: Optional[float] = None,
                 dpo_use_weighting: Optional[bool] = None,
                 dpo_use_logits_to_keep: Optional[bool] = None,
                 dpo_label_smoothing: Optional[float] = None,
                 dpo_norm_loss: Optional[bool] = None,
                 dpo_padding_free: Optional[bool] = None,
                 dpo_generate_during_eval: Optional[bool] = None,
                 dataset_type: str = "chatml.argilla",
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.trl_beta = trl_beta
        self.dpo_use_weighting = dpo_use_weighting
        self.dpo_use_logits_to_keep = dpo_use_logits_to_keep
        self.dpo_label_smoothing = dpo_label_smoothing
        self.dpo_norm_loss = dpo_norm_loss
        self.dpo_padding_free = dpo_padding_free
        self.dpo_generate_during_eval = dpo_generate_during_eval
        self.dataset_type = dataset_type

    def build(self) -> Dict[str, Any]:
        cfg = self.build_base()
        # Map to the Argilla-style pairwise format per Axolotl RLHF docs
        cfg["datasets"][0].update({
            "type": self.dataset_type,
        })
        # RL selector
        cfg["rl"] = "dpo"
        # TRL block (DPO uses beta)
        trl: Dict[str, Any] = {}
        if self.trl_beta is not None:
            trl["beta"] = float(self.trl_beta)
        if trl:
            cfg["trl"] = trl
        # DPO-specific toggles surfaced by Axolotl
        if self.dpo_use_weighting is not None:
            cfg["dpo_use_weighting"] = bool(self.dpo_use_weighting)
        if self.dpo_use_logits_to_keep is not None:
            cfg["dpo_use_logits_to_keep"] = bool(self.dpo_use_logits_to_keep)
        if self.dpo_label_smoothing is not None:
            cfg["dpo_label_smoothing"] = float(self.dpo_label_smoothing)
        if self.dpo_norm_loss is not None:
            cfg["dpo_norm_loss"] = bool(self.dpo_norm_loss)
        if self.dpo_padding_free is not None:
            cfg["dpo_padding_free"] = bool(self.dpo_padding_free)
        if self.dpo_generate_during_eval is not None:
            cfg["dpo_generate_during_eval"] = bool(self.dpo_generate_during_eval)
        return cfg
    
# ---------------------------------------------------------------------------
# PPO builder
# ---------------------------------------------------------------------------
class PPOConfigBuilder(BaseConfigBuilder):
    def __init__(self, *args,
                 # PPO hyperparameters
                 ppo_batch_size: int = 64,
                 ppo_mini_batch_size: int = 16,
                 ppo_epochs: int = 4,
                 ppo_learning_rate: float = 5e-6,
                 ppo_target_kl: float = 0.1,
                 ppo_kl_penalty: str = "kl",
                 ppo_cliprange: float = 0.2,
                 ppo_cliprange_value: float = 0.2,
                 ppo_gradient_accumulation_steps: int | None = None,
                 ppo_seed: int | None = 42,
                 ppo_ref_model: str | None = None,
                 # Generation parameters
                 gen_max_new_tokens: int = 256,
                 gen_temperature: float = 0.7,
                 gen_top_p: float = 0.9,
                 gen_top_k: int | None = None,
                 gen_do_sample: bool = True,
                 # Rewards
                 reward_funcs: list[str] | None = None,
                 reward_weights: list[float] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # PPO
        self.ppo_batch_size = ppo_batch_size
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.ppo_learning_rate = ppo_learning_rate
        self.ppo_target_kl = ppo_target_kl
        self.ppo_kl_penalty = ppo_kl_penalty
        self.ppo_cliprange = ppo_cliprange
        self.ppo_cliprange_value = ppo_cliprange_value
        self.ppo_gradient_accumulation_steps = ppo_gradient_accumulation_steps
        self.ppo_seed = ppo_seed
        self.ppo_ref_model = ppo_ref_model
        # Generation
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_top_k = gen_top_k
        self.gen_do_sample = gen_do_sample
        # Rewards
        self.reward_funcs = reward_funcs or [
            "rewards.model_helpfulness_reward",
            "rewards.think_format_reward",
        ]
        self.reward_weights = reward_weights or [1.0, 0.2]

    def build(self) -> Dict[str, Any]:
        cfg = self.build_base()
        # PPO dataset transform: emit {"query": ...}
        cfg["datasets"] = [{
            "path": self.dataset_path,
            "type": "rewards.messages_to_query_transform",
            "ppo": True
        }]

        cfg["ppo"] = {
            "batch_size": int(self.ppo_batch_size),
            "mini_batch_size": int(self.ppo_mini_batch_size),
            "ppo_epochs": int(self.ppo_epochs),
            "learning_rate": float(self.ppo_learning_rate),
            "target_kl": float(self.ppo_target_kl),
            "kl_penalty": str(self.ppo_kl_penalty),
            "cliprange": float(self.ppo_cliprange),
            "cliprange_value": float(self.ppo_cliprange_value),
            "gradient_accumulation_steps": int(self.ppo_gradient_accumulation_steps
                                               if self.ppo_gradient_accumulation_steps is not None
                                               else self.gradient_accumulation_steps),
            "seed": self.ppo_seed,
            "ref_model": self.ppo_ref_model,
        }

        cfg["generate"] = {
            "max_new_tokens": int(self.gen_max_new_tokens),
            "temperature": float(self.gen_temperature),
            "top_p": float(self.gen_top_p),
            "top_k": (None if self.gen_top_k is None else int(self.gen_top_k)),
            "do_sample": bool(self.gen_do_sample),
        }

        cfg["rewards"] = {
            "reward_funcs": list(self.reward_funcs),
            "reward_weights": [float(w) for w in self.reward_weights],
        }

        cfg["output_dir"] = self.output_dir
        return cfg

# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Axolotl/TRL YAML configs (SFT, DPO, GRPO, PPO).")
    sub = p.add_subparsers(dest="mode", required=True)

    # Common args function
    def add_common(sp: argparse.ArgumentParser):
        sp.add_argument("--base_model", required=True)
        sp.add_argument("--dataset_path", required=True)
        sp.add_argument("--output_path", required=True)
        sp.add_argument("--output_dir", default="./outputs/run")
        sp.add_argument("--sequence_len", type=int, default=4096)
        sp.add_argument("--micro_batch_size", type=int, default=1)
        sp.add_argument("--gradient_accumulation_steps", type=int, default=1)
        sp.add_argument("--num_epochs", type=int, default=3)
        sp.add_argument("--learning_rate", type=float, default=2e-4)
        sp.add_argument("--warmup_ratio", type=float, default=0.03)
        sp.add_argument("--val_set_size", type=float, default=0.02)
        sp.add_argument("--chat_template", default=None)
        sp.add_argument("--model_key", default=None)
        sp.add_argument("--resume_from_checkpoint", default=None)
        sp.add_argument("--auto_resume_from_checkpoints", action="store_true")
        # W&B
        sp.add_argument("--wandb_project", default=None)
        sp.add_argument("--wandb_entity", default=None)
        sp.add_argument("--wandb_name", default=None)
        sp.add_argument("--wandb_run_id", default=None)
        sp.add_argument("--wandb_mode", default=None)
        sp.add_argument("--wandb_watch", default=None)
        sp.add_argument("--wandb_log_model", default=None)
        sp.add_argument("--wandb_tags", default=None, help="comma-separated list")
        # Adapters (shared)
        sp.add_argument("--adapter", choices=["full", "lora", "qlora"], default="full")
        sp.add_argument("--lora_r", type=int, default=16)
        sp.add_argument("--lora_alpha", type=int, default=16)
        sp.add_argument("--lora_dropout", type=float, default=0.05)
        sp.add_argument("--lora_target_modules", default=None,
                        help="comma-separated list; overrides model defaults")

    # SFT subcommand
    sp_sft = sub.add_parser("sft", help="Generate SFT config")
    add_common(sp_sft)

    # DPO subcommand
    sp_dpo = sub.add_parser("dpo", help="Generate DPO config")
    add_common(sp_dpo)
    sp_dpo.add_argument("--trl_beta", type=float, default=None, help="TRL beta (temperature) for DPO loss")
    sp_dpo.add_argument("--dpo_use_weighting", action="store_true")
    sp_dpo.add_argument("--dpo_use_logits_to_keep", action="store_true")
    sp_dpo.add_argument("--dpo_label_smoothing", type=float, default=None)
    sp_dpo.add_argument("--dpo_norm_loss", action="store_true")
    sp_dpo.add_argument("--dpo_padding_free", action="store_true")
    sp_dpo.add_argument("--dpo_generate_during_eval", action="store_true")
    sp_dpo.add_argument("--dpo_dataset_type", default="chatml.argilla",
                        help="One of the DPO types in docs (e.g., chatml.argilla, chatml.intel, chat_template.default, user_defined.default)")

    # GRPO subcommand
    sp_grpo = sub.add_parser("grpo", help="Generate GRPO config")
    add_common(sp_grpo)
    sp_grpo.add_argument("--trl_num_generations", type=int, default=4)
    sp_grpo.add_argument("--trl_max_completion_length", type=int, default=256)
    sp_grpo.add_argument("--trl_reward_funcs", default=None,
                         help="comma-separated import paths to reward functions")
    sp_grpo.add_argument("--trl_reward_weights", default=None,
                         help="comma-separated floats, one per reward func")
    sp_grpo.add_argument("--trl_use_vllm", action="store_true")
    sp_grpo.add_argument("--trl_loss_type", default=None, help="e.g., dr_grpo")

    # PPO subcommand (YAML consumed by standalone TRL runner)
    sp_ppo = sub.add_parser("ppo", help="Generate PPO config for TRL runner")
    add_common(sp_ppo)
    # PPO hyperparameters
    sp_ppo.add_argument("--ppo_batch_size", type=int, default=64)
    sp_ppo.add_argument("--ppo_mini_batch_size", type=int, default=16)
    sp_ppo.add_argument("--ppo_epochs", type=int, default=4)
    sp_ppo.add_argument("--ppo_learning_rate", type=float, default=5e-6)
    sp_ppo.add_argument("--ppo_target_kl", type=float, default=0.1)
    sp_ppo.add_argument("--ppo_kl_penalty", default="kl")
    sp_ppo.add_argument("--ppo_cliprange", type=float, default=0.2)
    sp_ppo.add_argument("--ppo_cliprange_value", type=float, default=0.2)
    sp_ppo.add_argument("--ppo_gradient_accumulation_steps", type=int, default=None)
    sp_ppo.add_argument("--ppo_seed", type=int, default=42)
    sp_ppo.add_argument("--ppo_ref_model", default=None)
    # Generation
    sp_ppo.add_argument("--gen_max_new_tokens", type=int, default=256)
    sp_ppo.add_argument("--gen_temperature", type=float, default=0.7)
    sp_ppo.add_argument("--gen_top_p", type=float, default=0.9)
    sp_ppo.add_argument("--gen_top_k", type=int, default=None)
    sp_ppo.add_argument("--gen_do_sample", dest="gen_do_sample", action="store_true")
    sp_ppo.add_argument("--no_gen_do_sample", dest="gen_do_sample", action="store_false")
    sp_ppo.set_defaults(gen_do_sample=True)
    # Rewards
    sp_ppo.add_argument("--reward_funcs", default=None,
                        help="comma-separated import paths to reward functions")
    sp_ppo.add_argument("--reward_weights", default=None,
                        help="comma-separated floats, one per reward func")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_key = resolve_model_key(args.base_model, args.model_key)
    wandb_tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None

    common_kw = dict(
        base_model=args.base_model,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        sequence_len=args.sequence_len,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        val_set_size=args.val_set_size,
        chat_template=args.chat_template,
        model_key=model_key,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_from_checkpoints=getattr(args, "auto_resume_from_checkpoints", False),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
        wandb_watch=args.wandb_watch,
        wandb_log_model=args.wandb_log_model,
        wandb_tags=wandb_tags,
        adapter=args.adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=[x.strip() for x in args.lora_target_modules.split(",")] if getattr(args, "lora_target_modules", None) else None,
    )

    if args.mode == "sft":
        builder = SFTConfigBuilder(**common_kw)
        cfg = builder.build()
    elif args.mode == "dpo":
        builder = DPOConfigBuilder(
            **common_kw,
            trl_beta=getattr(args, "trl_beta", None),
            dpo_use_weighting=getattr(args, "dpo_use_weighting", None),
            dpo_use_logits_to_keep=getattr(args, "dpo_use_logits_to_keep", None),
            dpo_label_smoothing=getattr(args, "dpo_label_smoothing", None),
            dpo_norm_loss=getattr(args, "dpo_norm_loss", None),
            dpo_padding_free=getattr(args, "dpo_padding_free", None),
            dpo_generate_during_eval=getattr(args, "dpo_generate_during_eval", None),
            dataset_type=getattr(args, "dpo_dataset_type", "chatml.argilla"),
        )
        cfg = builder.build()
    else:  # grpo
        reward_funcs = None
        reward_weights = None
        if args.trl_reward_funcs:
            reward_funcs = [x.strip() for x in args.trl_reward_funcs.split(",") if x.strip()]
        if args.trl_reward_weights:
            reward_weights = [float(x.strip()) for x in args.trl_reward_weights.split(",") if x.strip()]
        builder = GRPOConfigBuilder(
            **common_kw,
            trl_num_generations=args.trl_num_generations,
            trl_max_completion_length=args.trl_max_completion_length,
            trl_reward_funcs=reward_funcs,
            trl_reward_weights=reward_weights,
            trl_use_vllm=getattr(args, "trl_use_vllm", False),
            trl_loss_type=args.trl_loss_type,
        )
        cfg = builder.build()

    BaseConfigBuilder.write_yaml(cfg, args.output_path)
    print(f"Wrote Axolotl config to {args.output_path}")


if __name__ == "__main__":
    main()
