"""
Reward functions and dataset transforms for RLHF training with TRL-compatible trainers.

Currently implemented rewards:
  - model_helpfulness_reward: wraps a Hugging Face reward model (e.g., Nemotron-*Reward)
  - think_format_reward: a thin wrapper over TRL's think_format_reward (requires TRL >=0.9)

Dataset transforms:
  - messages_to_prompt_transform: converts {"messages": [...]} rows into a single `prompt` field
  - messages_to_query_transform : same as above, but emits `{"query": ...}` when PPO usage is desired

Notes
-----
- These utilities are generic and can be used with different RL algorithms that expect
  reward functions of the form `f(prompts, completions, **kwargs) -> list[float]`.
- The model-based reward is configurable via environment variables so you can swap
  different reward model checkpoints without changing code.

Environment variables (optional)
--------------------------------
RM_ID                : HF model id for the reward model (default: nvidia/Qwen-3-Nemotron-32B-Reward)
RM_DEVICE_MAP        : device map for accelerate/transformers (default: "auto")
RM_DTYPE             : torch dtype for the reward model [bf16|fp16|fp32] (default: bf16 if CUDA else fp32)
RM_BATCH_SIZE        : batch size used when scoring in the reward function (default: 2)
RM_FORMAT            : input format for RM scoring [nothink|plain] (default: nothink)
                       - nothink: appends " /nothink" to the user turn and prepends an empty
                         `<think>\n\n</think>\n\n` block to the assistant text before scoring.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, Union

import torch
from functools import lru_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    # Available in TRL (>=0.9); we expose a wrapper with the same signature
    from trl.rewards import think_format_reward as _trl_think_format_reward
except Exception:  # pragma: no cover
    _trl_think_format_reward = None

JSONLike = Dict[str, Any]
Messages = List[Dict[str, str]]

# ---------------------------
# Config via environment
# ---------------------------
_RM_ID = os.getenv("RM_ID", "nvidia/Qwen-3-Nemotron-32B-Reward")
_RM_DEVICE_MAP = os.getenv("RM_DEVICE_MAP", "auto")
_RM_BATCH_SIZE = int(os.getenv("RM_BATCH_SIZE", "2"))
# _RM_FORMAT = os.getenv("RM_FORMAT", "nothink").lower()  # {nothink, plain}
_RM_FORMAT_ENV = os.getenv("RM_FORMAT", "auto").lower()  # {auto, plain, nothink}

_dtype_env = os.getenv("RM_DTYPE", "auto").lower()
if _dtype_env in {"bf16", "bfloat16"}:
    _RM_DTYPE = torch.bfloat16
elif _dtype_env in {"fp16", "float16", "half"}:
    _RM_DTYPE = torch.float16
elif _dtype_env in {"fp32", "float32"}:
    _RM_DTYPE = torch.float32
else:  # auto
    _RM_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Choose format based on model id unless explicitly set
_NOTHINK_MODEL_HINTS = (
    "nemotron",            # NVIDIA Nemotron reward models
    "qwen-3-nemotron",     # explicit hint for Qwen-3 Nemotron RM
)

def _resolve_rm_format(rm_id: str, rm_format_env: str) -> str:
    if rm_format_env in {"plain", "nothink"}:
        return rm_format_env
    rid = rm_id.lower()
    return "nothink" if any(h in rid for h in _NOTHINK_MODEL_HINTS) else "plain"

_RM_FORMAT = _resolve_rm_format(_RM_ID, _RM_FORMAT_ENV)

# Lazy singletons for the reward model
_TOKENIZER = None
_RM = None

# Regex to strip chain-of-thought blocks if present
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


# ============================================================================
# Dataset Transforms
# ============================================================================

def messages_to_prompt_transform(cfg: Any, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """Create a transform that converts `messages` → `prompt` for RL trainers.

    The returned callable has signature: `transform_fn(example, tokenizer=None) -> dict`.
    Axolotl or external runners will call it on each dataset row. We:
      1) Drop any assistant turns from `messages` so the policy generates them on-policy
      2) Apply the tokenizer's chat template if provided, emitting a single string prompt
         that ends at the assistant header (add_generation_prompt=True)
      3) Pass through selected metadata columns used by reward functions

    Returns
    -------
    transform_fn : callable(example: dict, tokenizer: Optional[PreTrainedTokenizer]) -> dict
    dataset_kwargs : dict with optional keys like {"remove_columns": ["messages"]}
    """
    passthrough_keys = (
        "answer_key",
        "reference_solution",
        "options",
        "original_user_content",
        "source_file",
    )

    def transform_fn(example: JSONLike, tokenizer=None) -> JSONLike:
        msgs: Messages = example.get("messages", []) or []
        pruned: Messages = [m for m in msgs if m.get("role") != "assistant"]

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                pruned,
                add_generation_prompt=True,  # end at assistant header
                tokenize=False,
            )
            out_prompt: Union[str, Messages] = prompt_text
        else:
            out_prompt = pruned  # conversational fallback

        out: JSONLike = {"prompt": out_prompt}
        for k in passthrough_keys:
            if k in example:
                out[k] = example[k]
        return out

    return transform_fn, {"remove_columns": ["messages"]}


def messages_to_query_transform(cfg: Any,
                                ppo: bool = True,
                                return_field: str | None = None,
                                remove_columns: List[str] | None = None
                                ) -> Tuple[Any, Dict[str, Any]]:
    """Create a transform that converts `messages` → `query` (for PPO) or `prompt`.

    Parameters
    ----------
    ppo : bool
        If True, output `{"query": ...}`. If False, behave like messages_to_prompt_transform().
    return_field : Optional[str]
        Override the output field name; if provided, ignores `ppo`.
    remove_columns : Optional[List[str]]
        Columns to drop after transform (default: ["messages"]).
    """
    if remove_columns is None:
        remove_columns = ["messages"]

    base_tf, _ = messages_to_prompt_transform(cfg)

    def transform_fn(example: JSONLike, tokenizer=None) -> JSONLike:
        out = base_tf(example, tokenizer=tokenizer)
        field = return_field if return_field is not None else ("query" if ppo else "prompt")
        if field == "prompt":
            return out
        # rename prompt → query
        return {"query": out.get("prompt")}
    return transform_fn, {"remove_columns": list(remove_columns)}


# ============================================================================
# Reward: Model-based helpfulness (Nemotron-like RMs)
# ============================================================================

def _load_rm() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load the reward model and tokenizer lazily as singletons."""
    global _TOKENIZER, _RM
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(_RM_ID, use_fast=True, trust_remote_code=True)
    # Ensure padding exists for batched scoring
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token or _TOKENIZER.sep_token or "<|endoftext|>"
    _TOKENIZER.padding_side = "right"

    if _RM is None:
        _RM = AutoModelForSequenceClassification.from_pretrained(
            _RM_ID, torch_dtype=_RM_DTYPE, device_map=_RM_DEVICE_MAP, trust_remote_code=True
        ).eval()

    # Mirror tokenizer pad id on the model to satisfy Transformers when batching
    _RM.config.pad_token_id = _TOKENIZER.pad_token_id
    return _TOKENIZER, _RM


def _extract_user_text_from_prompt(prompt: Union[str, Messages]) -> str:
    """From a string or list-of-messages prompt, extract the latest user text."""
    if isinstance(prompt, str):
        return prompt
    for msg in reversed(prompt):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return " ".join(m.get("content", "") for m in prompt)


def _normalize_completion_text(completion: Union[str, Messages]) -> str:
    """Return a plain string answer from the completion, stripping <think> blocks."""
    if isinstance(completion, list) and completion:
        text = completion[0].get("content", "")
    else:
        text = completion or ""
    return _THINK_RE.sub("", text).lstrip()


def model_helpfulness_reward(
    prompts: List[Union[str, Messages]],
    completions: List[Union[str, Messages]],
    **kwargs: Any,
) -> List[float]:
    """Model-based helpfulness reward scored by a HF reward model.

    Uses the RM_ID checkpoint. Builds a two-message conversation per sample and feeds it
    to the reward model. By default uses the "nothink" recipe:
      - append " /nothink" to the user query
      - prepend an empty `<think>\n\n</think>\n\n` block to the assistant answer
    """
    tok, rm = _load_rm()

    batch_messages: List[Messages] = []
    for p, c in zip(prompts, completions):
        user_text = _extract_user_text_from_prompt(p)
        ans_text = _normalize_completion_text(c)
        if _RM_FORMAT == "nothink":
            user_text = f"{user_text} /nothink"
            ans_text = "<think>\n\n</think>\n\n" + ans_text
        batch_messages.append([
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ans_text},
        ])
        # else: "plain", no special formatting

    scores: List[float] = []
    bs = max(1, _RM_BATCH_SIZE)
    for i in range(0, len(batch_messages), bs):
        chunk = batch_messages[i: i + bs]
        if hasattr(tok, "apply_chat_template"):
            toks = tok.apply_chat_template(
                chunk,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = toks["input_ids"].to(rm.device)
            attn = toks.get("attention_mask")
            attn = attn.to(rm.device) if attn is not None else None
        else:
            texts = [f"User: {m[0]['content']}\nAssistant: {m[1]['content']}" for m in chunk]
            toks = tok(texts, return_tensors="pt", padding=True, truncation=True)
            input_ids = toks["input_ids"].to(rm.device)
            attn = toks.get("attention_mask")
            attn = attn.to(rm.device) if attn is not None else None

        with torch.no_grad():
            logits = rm(input_ids=input_ids, attention_mask=attn).logits

        if logits.ndim == 2 and logits.size(-1) == 1:
            chunk_scores = logits.squeeze(-1)
        elif logits.ndim == 1:
            chunk_scores = logits
        else:
            chunk_scores = logits[..., -1]
        scores.extend([float(x) for x in chunk_scores.detach().cpu().tolist()])
    return scores


# ============================================================================
# Reward: Think-format compliance (wrapper around TRL's utility)
# ============================================================================

def think_format_reward(
    prompts: List[Union[str, Messages]],
    completions: List[Union[str, Messages]],
    **kwargs: Any,
) -> List[float]:
    """Reward that checks for `<think>...</think>` followed by a final answer."""
    if _trl_think_format_reward is None:
        raise ValueError("think_format_reward requires TRL >=0.9")
    # TRL expects: completions = List[List[{"content": str}]]
    trl_completions: List[List[Dict[str, str]]] = []
    for c in completions:
        if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
            trl_completions.append([{"content": c[0]["content"]}])
        else:
            text = str(c) if c is not None else ""
            trl_completions.append([{"content": text}])
    return [float(x) for x in _trl_think_format_reward(completions=trl_completions, **kwargs)]


__all__ = [
    "messages_to_prompt_transform",
    "messages_to_query_transform",
    "model_helpfulness_reward",
    "think_format_reward",
]
