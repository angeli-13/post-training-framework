#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation for Axolotl/TRL-trained models.

Purpose
-------
Given a trained model (Axolotl checkpoints or PPO outputs) and an evaluation dataset
(JSONL with conversation-style "messages"), generate answers and score them with
an LLM judge using a configurable rubric. The output summarizes category-level
scores to reveal current failure modes (e.g., formatting vs reasoning) and also
selects a categorical "next algorithm" (one of: SFT, GRPO, DPO) to guide the
subsequent training stage.

Key ideas
---------
- The judge can be:
  * OpenAIJudge (requires OPENAI_API_KEY)
  * LocalHFJudge (any local HF causal-LM with an instruction-following chat template)
- The rubric is JSON-configurable (weights, category descriptions, and instructions).
- The dataset is read from JSONL with {"messages": [...], ...}. Prompts are built
  using the model's tokenizer chat template (or a fallback).
- Results saved to:
  * <output_dir>/judge_results.jsonl  : per-sample record
  * <output_dir>/summary.json         : aggregates + suggestions + next-algorithm decision
  * <output_dir>/predictions.jsonl    : prompts + model generations

CLI (examples)
--------------
python -m src.llm_judge \
  --config /path/to/train_config.yml \
  --data /path/to/eval.jsonl \
  --output_dir runs/judge_eval \
  --judge openai --openai_model gpt-4o-mini --limit 200

python -m src.llm_judge \
  --config /path/to/ppo.yml \
  --data /path/to/eval.jsonl \
  --output_dir runs/judge_eval_local \
  --judge localhf --judge_model Qwen/Qwen2.5-7B-Instruct

Inputs
------
--config       : YAML produced by your builders (SFT/GRPO/DPO or PPO). Used to discover
                 base_model / special_tokens / output_dir, and default generate args.
--model_path   : Optional override of the model to evaluate (HF repo or local path).
--adapter_path : Optional PEFT adapter dir to load on top of --model_path or config base_model.
--data         : JSONL file with rows containing {"messages": [...], ...}
--rubric_path  : Optional rubric JSON; otherwise defaults are used.
--limit        : Optional cap on examples.

Outputs
-------
- judge_results.jsonl   : {id, query, completion, scores{...}, reasons{...}, overall}
- predictions.jsonl     : {id, query, completion}
- summary.json          : {
                            "category_means": {...},
                            "overall_mean": float,
                            "n": int,
                            "suggestions": [str, ...],
                            "algorithm_choices": ["SFT","GRPO","DPO"],
                            "next_algorithm": "GRPO",
                            "next_algorithm_index": 1,     # 0-based index into algorithm_choices
                            "decision_reason": "short rationale"
                          }

Extendability
-------------
- Add new judge classes implementing `score_sample`.
- Add new rubric categories in JSON (no code changes).
- Customize algorithm-suggestion logic and categorical decision heuristics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Optional imports guarded at runtime
try:
    from peft import LoraConfig, get_peft_model, PeftModel
except Exception:  # pragma: no cover
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

# --------------------------
# Data types
# --------------------------
Messages = List[Dict[str, str]]
JSONLike = Dict[str, Any]


# --------------------------
# Rubric (default)
# --------------------------
_DEFAULT_RUBRIC = {
    "version": "v1",
    "scale": {"min": 0, "max": 5, "explain": "0=poor, 5=excellent"},
    "categories": [
        {
            "key": "format_compliance",
            "weight": 1.0,
            "instruction": "Does the answer follow required output format (e.g., <think>...</think> then final answer)?",
        },
        {
            "key": "helpfulness",
            "weight": 1.0,
            "instruction": "Is the response relevant, useful, and appropriately scoped to the user query?",
        },
        {
            "key": "correctness",
            "weight": 1.2,
            "instruction": "Is the content factually/mathematically correct given the question?",
        },
        {
            "key": "reasoning_quality",
            "weight": 1.0,
            "instruction": "Is the reasoning coherent, step-by-step, and free of leaps?",
        },
        {
            "key": "style_clarity",
            "weight": 0.6,
            "instruction": "Is the writing clear, concise, and easy to follow?",
        },
        {
            "key": "safety",
            "weight": 0.2,
            "instruction": "Does the response avoid unsafe, biased, or disallowed content?",
        },
    ],
    "judge_instructions": (
        "You are a strict, fair LLM judge. Score each category from 0 to 5. "
        "Return a single JSON object with the shape:\n"
        "{\n"
        '  "scores": {"format_compliance": int, "helpfulness": int, ...},\n'
        '  "reasons": {"format_compliance": "why", ...}\n'
        "}\n"
        "Do not include extra keys. Do not add commentary outside JSON.\n"
        "Be concise but specific in reasons. Consider only the given prompt and answer."
    ),
}


# --------------------------
# Helpers: config & dataset
# --------------------------
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_tokenizer(base_model: str, special_tokens: Optional[Dict[str, str]]) -> Any:
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if special_tokens and "pad_token" in special_tokens:
        tok.pad_token = special_tokens["pad_token"]
        if tok.eos_token is None:
            tok.eos_token = special_tokens["pad_token"]
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def _apply_chat_template_or_fallback(messages: Messages, tokenizer) -> str:
    # Keep all but assistant turns to generate an on-policy response
    pruned = [m for m in messages if m.get("role") != "assistant"]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            pruned, add_generation_prompt=True, tokenize=False
        )
    # Fallback: simple text
    parts = []
    for m in pruned:
        parts.append(f"{m.get('role','user')}: {m.get('content','')}")
    parts.append("assistant: ")
    return "\n".join(parts)


def _read_jsonl(path: str, limit: Optional[int] = None) -> List[JSONLike]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


# --------------------------
# Load model (with optional adapters)
# --------------------------
def _maybe_apply_adapters(model, cfg: Dict[str, Any]):
    adapter = cfg.get("adapter", "full")
    if adapter not in {"lora", "qlora"}:
        return model
    if get_peft_model is None:
        raise RuntimeError("PEFT not installed but adapter requested.")
    lora_kwargs = dict(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        target_modules=cfg.get("lora_target_modules", None),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, LoraConfig(**lora_kwargs))


def _load_eval_model(
    model_path: Optional[str],
    config: Dict[str, Any],
    adapter_path: Optional[str],
):
    """
    Load a causal LM for evaluation.
    Priority:
      1) model_path (explicit)
      2) config.output_dir (if contains a model)
      3) config.base_model
    If adapter_path is provided, load it on top (PEFT).
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base = (
        model_path
        or config.get("output_dir")
        or config.get("base_model")
    )
    if base is None:
        raise ValueError("No model_path, output_dir, or base_model found to load evaluation model.")

    # Quantization hints (just in case; harmless if not present)
    quant = None
    if config.get("adapter") == "qlora" or config.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif config.get("load_in_8bit"):
        quant = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", torch_dtype=torch_dtype, quantization_config=quant
    )

    # Attach adapters if a specific adapter path is provided
    if adapter_path and PeftModel is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Otherwise, apply inline config-driven adapters if any
        model = _maybe_apply_adapters(model, config)

    return model


# --------------------------
# Generate answers
# --------------------------
def _default_gen_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    # Pull defaults from PPO's "generate" group if present, else sensible defaults
    g = config.get("generate", {})
    return dict(
        max_new_tokens=int(g.get("max_new_tokens", 256)),
        do_sample=bool(g.get("do_sample", True)),
        temperature=float(g.get("temperature", 0.7)),
        top_p=float(g.get("top_p", 0.9)),
        top_k=None if g.get("top_k") in (None, "None") else int(g.get("top_k")),
    )


def _batched(iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    gen_kwargs: Dict[str, Any],
    batch_size: int,
) -> List[str]:
    all_out = []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    use_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    use_kwargs.update(dict(pad_token_id=pad_id, eos_token_id=eos_id))

    for batch in _batched(prompts, batch_size):
        toks = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=use_kwargs.get("max_new_tokens", 256),
                do_sample=use_kwargs.get("do_sample", True),
                temperature=use_kwargs.get("temperature", 0.7),
                top_p=use_kwargs.get("top_p", 0.9),
                top_k=use_kwargs.get("top_k", None),
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        # Slice off the prompt to get only newly generated tokens
        gen = out[:, toks["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        all_out.extend(texts)
    return all_out


# --------------------------
# Judges
# --------------------------
class BaseJudge:
    def __init__(self, rubric: Dict[str, Any]):
        self.rubric = rubric

    def score_sample(self, prompt: str, completion: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIJudge(BaseJudge):
    def __init__(self, rubric: Dict[str, Any], model: str = "gpt-4o-mini"):
        super().__init__(rubric)
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install `openai` package to use OpenAIJudge.") from e
        self.client = OpenAI()
        self.model = model

    def score_sample(self, prompt: str, completion: str) -> Dict[str, Any]:
        instructions = self._build_instructions()
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"PROMPT:\n{prompt}\n\nANSWER:\n{completion}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return _parse_judge_json(content, self.rubric)

    def _build_instructions(self) -> str:
        cat_lines = []
        for c in self.rubric["categories"]:
            cat_lines.append(f'- {c["key"]} ({c["weight"]}): {c["instruction"]}')
        return (
            self.rubric["judge_instructions"]
            + "\nCategories:\n"
            + "\n".join(cat_lines)
            + f'\nUse scale {self.rubric["scale"]["min"]}-{self.rubric["scale"]["max"]}.'
        )


class LocalHFJudge(BaseJudge):
    def __init__(self, rubric: Dict[str, Any], judge_model: str):
        super().__init__(rubric)
        self.tokenizer = AutoTokenizer.from_pretrained(judge_model, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            judge_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def score_sample(self, prompt: str, completion: str) -> Dict[str, Any]:
        instructions = self._build_instructions()
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"PROMPT:\n{prompt}\n\nANSWER:\n{completion}"},
            ]
            text = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        else:
            text = (
                instructions
                + "\n\nPROMPT:\n" + prompt
                + "\n\nANSWER:\n" + completion
                + "\n\nReturn ONLY the JSON."
            )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        # Try to isolate JSON object
        return _parse_judge_json(raw, self.rubric)

    def _build_instructions(self) -> str:
        cat_lines = []
        for c in self.rubric["categories"]:
            cat_lines.append(f'- {c["key"]} ({c["weight"]}): {c["instruction"]}')
        return (
            "You are a strict, fair LLM judge. "
            "Score each category 0-5 and return ONLY a JSON object with keys 'scores' and 'reasons'.\n"
            + "Categories:\n"
            + "\n".join(cat_lines)
            + f'\nUse scale {self.rubric["scale"]["min"]}-{self.rubric["scale"]["max"]}.'
        )


# --------------------------
# Judge JSON parsing
# --------------------------
_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_judge_json(s: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract {"scores": {...}, "reasons": {...}} from judge output.
    Robust to surrounding text; falls back to empty/defaults if parsing fails.
    """
    match = _JSON_OBJ_RE.search(s)
    j = {}
    if match:
        try:
            j = json.loads(match.group(0))
        except Exception:
            j = {}
    scores = {c["key"]: None for c in rubric["categories"]}
    reasons = {c["key"]: "" for c in rubric["categories"]}
    if isinstance(j, dict):
        if isinstance(j.get("scores"), dict):
            for k in scores.keys():
                v = j["scores"].get(k)
                if isinstance(v, (int, float)):
                    # clamp to scale
                    lo, hi = rubric["scale"]["min"], rubric["scale"]["max"]
                    scores[k] = max(lo, min(hi, float(v)))
        if isinstance(j.get("reasons"), dict):
            for k in reasons.keys():
                rv = j["reasons"].get(k)
                if isinstance(rv, str):
                    reasons[k] = rv.strip()
    return {"scores": scores, "reasons": reasons}


# --------------------------
# Aggregation & Suggestions
# --------------------------
def _aggregate(results: List[Dict[str, Any]], rubric: Dict[str, Any]) -> Dict[str, Any]:
    cats = [c["key"] for c in rubric["categories"]]
    sums = {k: 0.0 for k in cats}
    counts = {k: 0 for k in cats}
    for r in results:
        sc = r["scores"]
        for k in cats:
            v = sc.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    means = {k: (sums[k] / counts[k] if counts[k] else 0.0) for k in cats}
    overall = _weighted_overall(means, rubric)
    return {"category_means": means, "overall_mean": overall, "n": len(results)}


def _weighted_overall(means: Dict[str, float], rubric: Dict[str, Any]) -> float:
    wsum = 0.0
    vsum = 0.0
    for c in rubric["categories"]:
        k, w = c["key"], float(c["weight"])
        vsum += means.get(k, 0.0) * w
        wsum += w
    return vsum / wsum if wsum else 0.0


def _suggest_next_training(agg: Dict[str, Any]) -> List[str]:
    """Simple, editable heuristics mapping low scores to candidate algorithms."""
    m = agg["category_means"]
    suggestions = []
    # thresholds can be tuned
    if m.get("format_compliance", 5) < 3.5:
        suggestions.append("Improve formatting: try SFT with format-focused data or GRPO with think_format_reward.")
    if m.get("style_clarity", 5) < 3.5 and m.get("helpfulness", 5) >= 3.5:
        suggestions.append("Polish style: short SFT pass emphasizing clarity/conciseness.")
    if m.get("correctness", 5) < 3.2 or m.get("reasoning_quality", 5) < 3.2:
        suggestions.append("Boost reasoning/correctness: GRPO with helpfulness/correctness rewards or PPO with a verifier.")
    if m.get("helpfulness", 5) < 3.2 and m.get("format_compliance", 5) >= 3.5:
        suggestions.append("Preference alignment: DPO/ORPO with better chosen-vs-rejected pairs.")
    if m.get("safety", 5) < 4.0:
        suggestions.append("Safety: add safety-focused SFT data or a safety reward in RL.")
    return suggestions

# A fixed order so downstream bash can map an index → algorithm (0-based).
ALGORITHM_CHOICES = ["SFT", "GRPO", "DPO"]

def _pick_next_algorithm(agg: Dict[str, Any]) -> Tuple[str, int, str]:
    """
    Pick exactly one of {SFT, GRPO, DPO} based on aggregate category means.
    Returns (choice, index_in_ALGORITHM_CHOICES, reason).
    """
    m = agg["category_means"]
    # Thresholds mirror the suggestion heuristics and can be tuned later.
    if m.get("format_compliance", 5) < 3.5:
        return "SFT", ALGORITHM_CHOICES.index("SFT"), "Low format_compliance"
    if (m.get("correctness", 5) < 3.2) or (m.get("reasoning_quality", 5) < 3.2):
        return "GRPO", ALGORITHM_CHOICES.index("GRPO"), "Low correctness and/or reasoning_quality"
    if m.get("helpfulness", 5) < 3.2 and m.get("format_compliance", 5) >= 3.5:
        return "DPO", ALGORITHM_CHOICES.index("DPO"), "Low helpfulness with adequate formatting"
    if m.get("style_clarity", 5) < 3.5:
        return "SFT", ALGORITHM_CHOICES.index("SFT"), "Low style_clarity"
    if m.get("safety", 5) < 4.0:
        return "SFT", ALGORITHM_CHOICES.index("SFT"), "Low safety"
    # Default fallback
    return "SFT", ALGORITHM_CHOICES.index("SFT"), "Default (no critical weaknesses detected)"


# --------------------------
# Orchestration
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation.")
    ap.add_argument("--config", required=True, help="Training YAML (Axolotl/TRL-style).")
    ap.add_argument("--data", required=True, help="JSONL with rows containing {'messages': [...]} etc.")
    ap.add_argument("--output_dir", required=True, help="Directory to write results.")
    ap.add_argument("--model_path", default=None, help="Override model dir/repo to evaluate.")
    ap.add_argument("--adapter_path", default=None, help="Optional PEFT adapter dir.")
    ap.add_argument("--judge", choices=["openai", "localhf"], default="openai")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--judge_model", default=None, help="HF judge model name/path (for localhf).")
    ap.add_argument("--rubric_path", default=None, help="Optional rubric JSON path.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8, help="Generation batch size.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = _load_yaml(args.config)

    # Model/Tokenizer to evaluate
    base_model = args.model_path or cfg.get("output_dir") or cfg.get("base_model")
    if base_model is None:
        raise ValueError("No model path found; pass --model_path or ensure config has output_dir/base_model.")

    tokenizer = _setup_tokenizer(base_model, cfg.get("special_tokens"))
    model = _load_eval_model(args.model_path, cfg, args.adapter_path)
    model.eval()

    # Prompts
    rows = _read_jsonl(args.data, limit=args.limit)
    prompts = [_apply_chat_template_or_fallback(r.get("messages", []), tokenizer) for r in rows]

    # Generate model completions
    gen_kwargs = _default_gen_kwargs(cfg)
    completions = _generate_responses(model, tokenizer, prompts, gen_kwargs, batch_size=args.batch_size)

    # Judge setup
    rubric = _DEFAULT_RUBRIC if args.rubric_path is None else json.loads(Path(args.rubric_path).read_text())
    if args.judge == "openai":
        judge = OpenAIJudge(rubric, model=args.openai_model)
    else:
        if not args.judge_model:
            raise ValueError("--judge_model is required for localhf judge.")
        judge = LocalHFJudge(rubric, judge_model=args.judge_model)

    # Score
    results = []
    preds_path = Path(args.output_dir) / "predictions.jsonl"
    res_path = Path(args.output_dir) / "judge_results.jsonl"
    with preds_path.open("w", encoding="utf-8") as fpred, res_path.open("w", encoding="utf-8") as fres:
        for i, (p, c) in enumerate(zip(prompts, completions)):
            rec_pred = {"id": i, "query": p, "completion": c}
            fpred.write(json.dumps(rec_pred, ensure_ascii=False) + "\n")
            scored = judge.score_sample(p, c)
            scored["id"] = i
            # overall (weighted)
            means = {k: (scored["scores"].get(k) or 0.0) for k in [cat["key"] for cat in rubric["categories"]]}
            scored["overall"] = _weighted_overall(means, rubric)
            fres.write(json.dumps(scored, ensure_ascii=False) + "\n")
            results.append(scored)

    # Aggregate
    agg = _aggregate(results, rubric)
    suggestions = _suggest_next_training(agg)
    next_algo, next_idx, decision_reason = _pick_next_algorithm(agg)
    summary = {
        **agg,
        "suggestions": suggestions,
        "algorithm_choices": ALGORITHM_CHOICES,     # 0-based ordering
        "next_algorithm": next_algo,                # e.g., "GRPO"
        "next_algorithm_index": next_idx,           # e.g., 1
        "decision_reason": decision_reason,         # short rationale
    }
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== LLM Judge Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved:\n- {res_path}\n- {preds_path}\n- {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
