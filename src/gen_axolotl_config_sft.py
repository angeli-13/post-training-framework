#!/usr/bin/env python3
"""
Generate an Axolotl YAML configuration for supervised fine-tuning (SFT).

This script focuses on chat-style datasets (OpenAI messages format) and
produces a config you can pass directly to:

    axolotl train <generated.yml>

What it does for you
- Sets sane defaults for Qwen3-4B-Instruct-2507
- Uses tokenizer chat template by default (keeps training + inference aligned)
- Enables LoRA or QLoRA when requested, or full fine-tuning
- Masks non-assistant turns during training
- Supports resuming from a checkpoint
- Supports Weights & Biases (W&B) project/run configuration

Example
-------
python gen_axolotl_config_sft.py \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset_path /data/my_conversations.jsonl \
  --output_path qwen3_sft.yml \
  --adapter qlora \
  --output_dir outputs/qwen3-sft \
  --sequence_len 8192 \
  --micro_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --wandb_project hhri-foxbrain \
  --wandb_name qwen3-sft-run1

Then run:
  axolotl train qwen3_sft.yml

Notes
-----
- Your dataset should be JSONL with entries like:
  {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
- The config will train only on assistant turns.
- If your base_model is a local path, pass --model_key to select which defaults to apply
  (or rely on auto-detection that matches known model names inside the path).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Per-model tweaks and safe defaults
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Qwen3-4B-Instruct-2507 specifics
    "Qwen/Qwen3-4B-Instruct-2507": {
        # Use tokenizer chat template shipped with the model
        "chat_template": "tokenizer_default",
        # Explicit special tokens for this model family (per HF tokenizer_config)
        "special_tokens": {
            "pad_token": "<|endoftext|>",
        },
        # Some Qwen variants rely on custom tokenizer classes in transformers
        "trust_remote_code": True,
        # A good starting set of LoRA targets for Qwen-family models
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    # Qwen3-4B-Thinking-2507 specifics (same tokenizer + tokens; thinking chat template)
    "Qwen/Qwen3-4B-Thinking-2507": {
        "chat_template": "tokenizer_default",
        "special_tokens": {
            "pad_token": "<|endoftext|>",
        },
        "trust_remote_code": True,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
}


def resolve_model_key(base_model: str, explicit: Optional[str]) -> Optional[str]:
    """Return a known model key for defaults.

    Tries, in order: explicit value, exact match, substring match on the
    last path component of known keys inside the provided base_model string.
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


def build_base_config(
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
) -> Dict[str, Any]:
    """Create the common portion of the Axolotl config.

    Parameters
    ----------
    base_model : str
        Hugging Face model id or local path.
    dataset_path : str
        Path to JSONL dataset in messages/role/content format.
    output_dir : str
        Directory for checkpoints and final artifacts.
    sequence_len : int
        Max packed sequence length used for training and eval.
    micro_batch_size : int
        Per-device micro batch size.
    gradient_accumulation_steps : int
        Accumulation steps to reach effective batch size.
    num_epochs : int
        Number of training epochs.
    learning_rate : float
        Optimizer learning rate (AdamW family).
    warmup_ratio : float
        Proportion of total steps used for warmup.
    val_set_size : float
        Fraction to hold out for validation.
    run_name : str | None
        Optional experiment name for tracking (W&B etc.).
    chat_template : str | None
        Optional override for chat template (defaults to tokenizer_default for known models).
    model_key : str | None
        Which model defaults to apply when base_model is a path.
    resume_from_checkpoint : str | None
        If provided, Axolotl will resume training from this checkpoint directory.
    auto_resume_from_checkpoints : bool | None
        If True and resume_from_checkpoint is not set, Axolotl will try to pick up the latest
        checkpoint in the output_dir automatically.
    W&B fields
        wandb_project, wandb_entity, wandb_name, wandb_run_id, wandb_mode, wandb_watch,
        wandb_log_model, wandb_tags
    """
    model_defaults = MODEL_DEFAULTS.get(model_key or base_model, {})

    config: Dict[str, Any] = {
        "base_model": base_model,
        # Keep tokenizer/model-specific behavior available when needed
        "trust_remote_code": bool(model_defaults.get("trust_remote_code", False)),

        # Use tokenizer chat template, or a given template like "chatml"/"gemma"/etc.
        "chat_template": chat_template or model_defaults.get("chat_template", "tokenizer_default"),

        # Dataset section: chat-style conversations, mask training to assistant turns
        "datasets": [
            {
                "path": dataset_path,
                "type": "chat_template",
                "roles_to_train": ["assistant"],
                # Train on EOS at each assistant turn boundary
                "train_on_eos": "turn",
            }
        ],
        # Cache pre-tokenized dataset here if you call `axolotl preprocess`
        "dataset_prepared_path": "last_run_prepared",

        # Eval split size if you are not supplying explicit test_datasets
        "val_set_size": val_set_size,

        # Training + packing
        "sequence_len": sequence_len,
        "sample_packing": True,
        "eval_sample_packing": False,
        "pad_to_sequence_len": True,

        # Core training hyperparameters
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "lr_scheduler": "cosine",
        "warmup_ratio": warmup_ratio,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},

        # dtype and output
        "bf16": True,
        "output_dir": output_dir,
    }

    # Attach model-specific special tokens when known
    if "special_tokens" in model_defaults:
        config["special_tokens"] = model_defaults["special_tokens"]

    # Run name for trackers like W&B (optional)
    if run_name:
        config["wandb_name"] = run_name

    # Checkpoint resume controls
    if resume_from_checkpoint:
        config["resume_from_checkpoint"] = str(resume_from_checkpoint)
    if auto_resume_from_checkpoints is not None:
        config["auto_resume_from_checkpoints"] = bool(auto_resume_from_checkpoints)

    # W&B block (added only if provided)
    if wandb_project:
        config["wandb_project"] = wandb_project
    if wandb_entity:
        config["wandb_entity"] = wandb_entity
    if wandb_name:
        config["wandb_name"] = wandb_name
    if wandb_run_id:
        config["wandb_run_id"] = wandb_run_id
    if wandb_mode:
        config["wandb_mode"] = wandb_mode
    if wandb_watch:
        config["wandb_watch"] = wandb_watch
    if wandb_log_model:
        config["wandb_log_model"] = wandb_log_model
    if wandb_tags:
        config["wandb_tags"] = list(wandb_tags)

    return config


def add_lora_blocks(
    cfg: Dict[str, Any],
    adapter: str,
    model_key: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> None:
    """Augment the config with LoRA/QLoRA settings.

    Parameters
    ----------
    cfg : dict
        Mutable config being built.
    adapter : str
        One of {"lora", "qlora"}. "full" means no adapters.
    lora_r, lora_alpha, lora_dropout : hyperparameters for PEFT LoRA.
    target_modules : list[str] | None
        Which linear modules to adapt. If None, uses model defaults when known.
    """
    adapter = adapter.lower()
    if adapter not in {"lora", "qlora"}:
        return

    cfg["adapter"] = adapter

    # Quantization choice
    if adapter == "qlora":
        cfg.update(
            {
                "load_in_4bit": True,
                # BitsAndBytes quantization recipe commonly used for QLoRA
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                # Optimizer that pairs well with 4-bit params
                "optimizer": "paged_adamw_8bit",
            }
        )
    else:  # classic LoRA
        cfg.update(
            {
                "load_in_8bit": True,
                "optimizer": "adamw_torch",
            }
        )

    # LoRA hyperparameters
    cfg.update(
        {
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
        }
    )

    # Target modules
    if target_modules is None:
        lookup_key = model_key or cfg["base_model"]
        target_modules = MODEL_DEFAULTS.get(lookup_key, {}).get("lora_target_modules")
    if target_modules:
        cfg["lora_target_modules"] = list(target_modules)


def write_yaml(config: Dict[str, Any], output_path: str) -> None:
    """Serialize the config as YAML with stable key ordering.

    Parameters
    ----------
    config : dict
        Final Axolotl configuration.
    output_path : str
        Path to write the YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an Axolotl SFT YAML config.")
    # Required I/O
    p.add_argument("--base_model", required=True, help="HF model id or local path")
    p.add_argument("--dataset_path", required=True, help="Path to JSONL dataset (messages format)")
    p.add_argument("--output_path", required=True, help="Where to write the YAML config")

    # Training outputs
    p.add_argument("--output_dir", default="./outputs/run", help="Axolotl output directory")
    p.add_argument("--run_name", default=None, help="Optional run name (e.g. for W&B)")

    # Packing + lengths
    p.add_argument("--sequence_len", type=int, default=4096, help="Packed sequence length")

    # Batching
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Schedule
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--val_set_size", type=float, default=0.02)

    # Adapters
    p.add_argument(
        "--adapter",
        choices=["full", "lora", "qlora"],
        default="qlora",
        help="Use full fine-tuning or parameter-efficient adapters",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated module names to adapt (overrides model defaults)",
    )

    # Prompt formatting
    p.add_argument(
        "--chat_template",
        default=None,
        help="Override chat template (default uses tokenizer_default)",
    )

    # Model defaults resolution
    p.add_argument(
        "--model_key",
        default=None,
        help=(
            "Name of the model whose defaults to apply (e.g., 'Qwen/Qwen3-4B-Instruct-2507'). "
            "Useful when --base_model is a local path. If omitted, the script will try to infer it."
        ),
    )

    # Resume options
    p.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Checkpoint directory to resume from (adds resume_from_checkpoint to YAML)",
    )
    p.add_argument(
        "--auto_resume_from_checkpoints",
        action="store_true",
        help="If set and --resume_from_checkpoint is not, auto-resume from latest checkpoint",
    )

    # Weights & Biases
    p.add_argument("--wandb_project", default=None, help="W&B project name")
    p.add_argument("--wandb_entity", default=None, help="W&B entity (team) name")
    p.add_argument("--wandb_name", default=None, help="W&B run name")
    p.add_argument("--wandb_run_id", default=None, help="W&B run id (to resume a run)")
    p.add_argument("--wandb_mode", default=None, help='W&B mode: "online", "offline", or "disabled"')
    p.add_argument("--wandb_watch", default=None, help="W&B watch setting (e.g., gradients")
    p.add_argument("--wandb_log_model", default=None, help="W&B log model policy (e.g., true|false|checkpoint)")
    p.add_argument(
        "--wandb_tags",
        default=None,
        help="Comma-separated W&B tags",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_key = resolve_model_key(args.base_model, args.model_key)

    tags_list = None
    if args.wandb_tags:
        tags_list = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    cfg = build_base_config(
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
        run_name=args.run_name,
        chat_template=args.chat_template,
        model_key=model_key,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_from_checkpoints=args.auto_resume_from_checkpoints,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_run_id=args.wandb_run_id,
        wandb_mode=args.wandb_mode,
        wandb_watch=args.wandb_watch,
        wandb_log_model=args.wandb_log_model,
        wandb_tags=tags_list,
    )

    target_modules = None
    if args.lora_target_modules:
        target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    add_lora_blocks(
        cfg,
        adapter=args.adapter,
        model_key=model_key,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    write_yaml(cfg, args.output_path)
    print(f"Wrote Axolotl config to {args.output_path}")


if __name__ == "__main__":
    main()
