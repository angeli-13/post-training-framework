# src/ppo_train.py
#!/usr/bin/env python3
"""
Standalone TRL PPO training driven by a YAML config (compatible with PPOConfigBuilder).

Design
------
This script reads a YAML file that mirrors the style of Axolotl configs and runs PPO using TRL.

YAML schema (top-level keys used here)
--------------------------------------
base_model: str                   # HF repo or local path for the policy model
chat_template: str                # usually "tokenizer_default" (optional but recommended)
special_tokens: {pad_token: str}  # ensures tokenizer.pad_token set correctly (recommended)
datasets:
  - path: str                     # JSONL file with {"messages": [...], ...}
    type: str                     # dotted path to a factory returning (transform_fn, dataset_kwargs)
                                  # e.g., "rewards.messages_to_query_transform"
    ppo: bool                     # set True so the transform emits {"query": ...} for PPO
output_dir: str

# Optional adapters (applied to policy model if present)
adapter: "full" | "lora" | "qlora"
lora_r: int
lora_alpha: int
lora_dropout: float
lora_target_modules: [str, ...]
# If adapter == "qlora", these may also appear (from your config builder)
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "bfloat16"

# PPO group
ppo:
  batch_size: int
  mini_batch_size: int
  ppo_epochs: int
  learning_rate: float
  target_kl: float
  kl_penalty: str                 # "kl", "abs", "mse", "full"
  cliprange: float
  cliprange_value: float
  gradient_accumulation_steps: int
  seed: int | null
  ref_model: str | null           # optional HF path for reference model (else, internal copy)

# Generation group
generate:
  max_new_tokens: int
  temperature: float
  top_p: float
  top_k: int | null
  do_sample: bool

# Rewards group
rewards:
  reward_funcs: [ "rewards.model_helpfulness_reward", ... ]
  reward_weights: [ 1.0, 0.2, ... ]

# Optional logging/resume (environment only for W&B; PPO stateful resume is not handled here)
wandb_project, wandb_entity, wandb_name, wandb_run_id, wandb_mode, wandb_tags
"""

from __future__ import annotations

import argparse
import json
import math
import os
from importlib import import_module
from typing import Any, Dict, List, Tuple

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    set_seed,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _import_callable(path: str):
    mod, fn = path.rsplit(".", 1)
    return getattr(import_module(mod), fn)

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _setup_tokenizer(base_model: str, special_tokens: Dict[str, str] | None) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    # Respect special pad token if provided
    if special_tokens and "pad_token" in special_tokens:
        tok.pad_token = special_tokens["pad_token"]
        if tok.eos_token is None:
            tok.eos_token = special_tokens["pad_token"]
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok

def _maybe_apply_adapters(model: AutoModelForCausalLM, cfg: Dict[str, Any]) -> AutoModelForCausalLM:
    adapter = cfg.get("adapter", "full")
    if adapter not in {"lora", "qlora"}:
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PEFT is required for LoRA/QLoRA adapters. Install `peft`.") from e

    lora_kwargs = dict(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=cfg.get("lora_target_modules", None),
        task_type="CAUSAL_LM",
    )
    peft_config = LoraConfig(**lora_kwargs)
    return get_peft_model(model, peft_config)

def _load_policy_and_ref_models(base_model: str, cfg: Dict[str, Any]):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    quant_config = None
    # If the YAML came from a QLoRA config, prefer 4-bit loading
    if cfg.get("adapter", "full") == "qlora" or cfg.get("load_in_4bit", False):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=getattr(torch, str(cfg.get("bnb_4bit_compute_dtype", "bfloat16"))),
        )
    elif cfg.get("load_in_8bit", False):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    policy_base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    policy_base = _maybe_apply_adapters(policy_base, cfg)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(policy_base)

    ref_path = (cfg.get("ppo") or {}).get("ref_model")
    if ref_path:
        ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            ref_path, device_map="auto", torch_dtype=torch_dtype
        )
    else:
        # TRL can internally handle a reference copy when ref_model=None
        ref = None
    return policy, ref

def _build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    p = cfg.get("ppo", {})
    return PPOConfig(
        batch_size=int(p.get("batch_size", 64)),
        mini_batch_size=int(p.get("mini_batch_size", 16)),
        ppo_epochs=int(p.get("ppo_epochs", 4)),
        learning_rate=float(p.get("learning_rate", 5e-6)),
        target_kl=float(p.get("target_kl", 0.1)),
        kl_penalty=str(p.get("kl_penalty", "kl")),
        cliprange=float(p.get("cliprange", 0.2)),
        cliprange_value=float(p.get("cliprange_value", 0.2)),
        gradient_accumulation_steps=int(p.get("gradient_accumulation_steps", 1)),
        seed=p.get("seed", 42),
        # Additional PPOConfig fields can be added as needed
    )

def _build_gen_kwargs(cfg: Dict[str, Any], tok: AutoTokenizer) -> Dict[str, Any]:
    g = cfg.get("generate", {})
    return dict(
        max_new_tokens=int(g.get("max_new_tokens", 256)),
        do_sample=bool(g.get("do_sample", True)),
        temperature=float(g.get("temperature", 0.7)),
        top_p=float(g.get("top_p", 0.9)),
        top_k=(None if g.get("top_k") in (None, "None") else int(g.get("top_k"))),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

def _load_reward_fns(cfg: Dict[str, Any]):
    r = cfg.get("rewards", {})
    fn_paths = r.get("reward_funcs", [])
    weights = r.get("reward_weights", [])
    if len(weights) and len(weights) != len(fn_paths):
        raise ValueError("reward_weights length must match reward_funcs length.")
    fns = [_import_callable(p) for p in fn_paths]
    wts = [float(w) for w in (weights or [1.0] * len(fns))]
    return fns, wts

def _compute_rewards(fns, wts, prompts: List[str], responses: List[str], batch_meta: Dict[str, Any]) -> List[float]:
    totals = [0.0] * len(responses)
    for fn, w in zip(fns, wts):
        scores = fn(prompts=prompts, completions=responses, **batch_meta)
        if len(scores) != len(responses):
            raise ValueError("Reward function returned wrong number of scores.")
        for i, s in enumerate(scores):
            totals[i] += float(w) * float(s)
    return totals

def _load_transform(dspec: Dict[str, Any]):
    """
    dspec['type'] is a dotted factory path returning (transform_fn, dataset_kwargs).
    Extra keys in dspec (e.g., 'ppo': true) are forwarded to the factory.
    """
    factory = _import_callable(dspec["type"])
    kwargs = {k: v for k, v in dspec.items() if k not in {"type", "path"}}
    return factory, kwargs

def _apply_transform_to_jsonl(jsonl_path: str, transform_factory, factory_kwargs, tokenizer) -> Dataset:
    """
    Reads a JSONL file of rows with {"messages": [...], ...} and converts it to a
    Dataset that contains 'query' strings for PPO.
    """
    transform_fn, dkw = transform_factory(cfg=None, **factory_kwargs)
    remove_cols = set(dkw.get("remove_columns", []))
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            out = transform_fn(ex, tokenizer=tokenizer)
            for rc in remove_cols:
                out.pop(rc, None)
            rows.append(out)
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run TRL PPO from a YAML config.")
    ap.add_argument("--config", required=True, help="Path to PPO YAML config file.")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    # W&B environment setup (optional)
    if cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = str(cfg["wandb_project"])
    if cfg.get("wandb_entity"):
        os.environ["WANDB_ENTITY"] = str(cfg["wandb_entity"])
    if cfg.get("wandb_mode"):
        os.environ["WANDB_MODE"] = str(cfg["wandb_mode"])
    if cfg.get("wandb_name"):
        os.environ["WANDB_NAME"] = str(cfg["wandb_name"])
    if cfg.get("wandb_tags"):
        os.environ["WANDB_TAGS"] = ",".join(cfg["wandb_tags"])

    # Tokenizer
    tok = _setup_tokenizer(cfg["base_model"], cfg.get("special_tokens"))

    # Dataset (expect a single spec for simplicity)
    dspec = cfg["datasets"][0]
    tf_factory, tf_kwargs = _load_transform(dspec)
    # Ensure PPO flag produces {"query": ...}
    tf_kwargs = {**tf_kwargs, "ppo": True}
    dataset = _apply_transform_to_jsonl(dspec["path"], tf_factory, tf_kwargs, tokenizer=tok)

    # Models and PPO trainer
    ppo_conf = _build_ppo_config(cfg)
    if ppo_conf.seed is not None:
        set_seed(int(ppo_conf.seed))

    policy, ref_model = _load_policy_and_ref_models(cfg["base_model"], cfg)
    ppo_trainer = PPOTrainer(
        ppo_conf,
        model=policy,
        ref_model=ref_model,
        tokenizer=tok,
        dataset=dataset,
    )

    gen_kwargs = _build_gen_kwargs(cfg, tok)
    reward_fns, reward_wts = _load_reward_fns(cfg)

    # Train for the configured number of epochs (iterate dataset each epoch)
    num_epochs = int(cfg.get("num_epochs", 1))
    for epoch in range(num_epochs):
        for batch in ppo_trainer.dataloader:
            queries: List[str] = batch["query"]
            # Generate responses
            response_tensors = ppo_trainer.generate(
                queries,
                **{k: v for k, v in gen_kwargs.items() if v is not None}
            )
            responses = tok.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute rewards and take a PPO step
            rewards = _compute_rewards(reward_fns, reward_wts, prompts=queries, responses=responses, batch_meta=batch)
            ppo_trainer.step(queries, responses, rewards)

        # Optional: save a checkpoint per epoch
        outdir = os.path.join(cfg["output_dir"], f"epoch-{epoch+1}")
        os.makedirs(outdir, exist_ok=True)
        ppo_trainer.save_pretrained(outdir)

    # Save final policy (and tokenizer) at output_dir
    final_out = cfg["output_dir"]
    os.makedirs(final_out, exist_ok=True)
    ppo_trainer.save_pretrained(final_out)
    tok.save_pretrained(final_out)
    print(f"PPO training complete. Saved to {final_out}")

if __name__ == "__main__":
    main()
