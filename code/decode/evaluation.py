"""Evaluation utilities for sequence models (WER, TER)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Any

import torch
from decode.decode import (greedy_decode, nbest_decode, build_id2sym,
                           ids_to_symbols)
from decode.inference import predict_logits


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Standard Levenshtein distance between two sequences using DP."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j - 1], dp[j], prev)
            prev = tmp
    return dp[m]


def compute_error_rate(references: list[Sequence], hypotheses: list[Sequence]) -> float:
    """Compute error rate (WER or TER) over a set of samples."""
    if not references: return 0.0
    total_dist = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    total_len = sum(len(r) for r in references)
    return total_dist / total_len if total_len > 0 else (1.0 if total_dist > 0 else 0.0)


def _align_dp(ref: Sequence, hyp: Sequence) -> list[list[int]]:
    """Full Levenshtein DP table (used by the backtracing helpers below)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp


def align_ops(ref: Sequence, hyp: Sequence) -> dict[str, int]:
    """Levenshtein alignment broken into operation counts.

    Returns ``{"S", "D", "I", "C", "total"}`` where S/D/I are substitution,
    deletion and insertion counts, C is the number of matched tokens and total
    is ``len(ref)``. The edit distance is ``S + D + I`` and the error rate against
    the reference is ``(S + D + I) / total``. Backtrace order (match, then
    substitution, then deletion, then insertion) matches ``_get_word_errors`` in
    adjust.py so the two stay consistent.
    """
    n, m = len(ref), len(hyp)
    dp = _align_dp(ref, hyp)
    S = D = I = C = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            C += 1; i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return {"S": S, "D": D, "I": I, "C": C, "total": n}


def aligned_tokens(ref: Sequence, hyp: Sequence) -> tuple[list, list]:
    """Backtrace returning ``(correct_ref_tokens, wrong_ref_tokens)``.

    A reference token is *correct* if it aligns to an identical hypothesis token
    (a match) and *wrong* if it is substituted or deleted. Insertions add no ref
    token to either list. Used for the per-word "which words can the system
    decode" analysis (system output vs ground truth).
    """
    n, m = len(ref), len(hyp)
    dp = _align_dp(ref, hyp)
    correct, wrong = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            correct.append(ref[i - 1]); i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            wrong.append(ref[i - 1]); i -= 1; j -= 1          # substitution
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            wrong.append(ref[i - 1]); i -= 1                  # deletion
        else:
            j -= 1                                            # insertion
    correct.reverse(); wrong.reverse()
    return correct, wrong


def evaluate_split(inference_out: dict[str, Any], decode_type: str = "greedy") -> dict[str, float]:
    """Evaluate a single split's output."""
    cfg = inference_out["config"]
    blank = cfg.get("training", {}).get("ctc_blank", 0)
    
    refs, hyps = [], []
    for batch in inference_out["results"]:
        if decode_type == "greedy":
            preds = greedy_decode(batch["logits"], lengths=batch["out_len"], blank=blank)
        else:
            raise ValueError(f"Unsupported decode_type: {decode_type}")
            
        for i in range(len(preds)):
            refs.append(batch["labels"][i, : int(batch["label_len"][i].item())].tolist())
            hyps.append(preds[i])
            
    err = compute_error_rate(refs, hyps)
    # [CENSORED] Evaluates Token Error Rate (TER) for intermediate units, or WER for words
    metric = "TER" if cfg.get("transform", {}).get("token_based", True) else "WER"
    return {metric: err}


def run_encoder_evaluation(
    run_dir: str | Path, 
    device: str = "auto", 
    checkpoint_name: str = "model.pt"
) -> dict[str, Any]:
    """Runs greedy evaluation on all splits and saves to encoder_eval.json."""
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / checkpoint_name
    
    results_summary = {"checkpoint": checkpoint_name}
    for split in ["train", "val", "test"]:
        print(f"Evaluating {split} ({checkpoint_name})...")
        try:
            inf_out = predict_logits(checkpoint_path, split=split, device=device)
            results_summary[split] = evaluate_split(inf_out, decode_type="greedy")
        except Exception as e:
            print(f"Could not evaluate {split}: {e}")
            
    eval_file = run_dir / f"encoder_eval_{checkpoint_name.replace('.pt', '')}.json"
    eval_file.write_text(json.dumps(results_summary, indent=2))
    return results_summary


def _seq_err(ref: Sequence, hyp: Sequence) -> float:
    """Per-utterance error rate: edit_distance / len(ref) (1.0 if ref empty but hyp not)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def dump_encoder_predictions(
    run_dir: str | Path,
    split: str = "test",
    *,
    n_best: int = 5,
    beam_size: int = 10,
    token_beam: int | None = None,
    checkpoint_name: str = "model_best.pt",
    device: str = "auto",
) -> Path:
    """Write one JSON record per utterance to ``encoder_predictions.jsonl`` or _val postfix .

    This is the Condition A (encoder-only) substrate for Experiment 1 analysis.
    It runs on EVERY run (intermediate token and word, no API cost). The CTC target units
    are intermediate tokens for a token run and words for a word run; ``unit`` records
    which. Per utterance it stores the ground-truth sentence and unit sequence,
    the greedy top-1, the n-best beam candidates, the per-candidate error rate,
    the greedy S/D/I/C breakdown and the oracle best-of-N error (the encoder
    ceiling the post-processor could in principle reach).
    """
    run_dir = Path(run_dir)
    inf = predict_logits(run_dir / checkpoint_name, split=split, device=device)
    cfg = inf["config"]
    is_token_branch = cfg.get("transform", {}).get("token_based", True)
    unit = "token" if is_token_branch else "word"
    blank_id = cfg.get("training", {}).get("ctc_blank", 0)
    id2sym = build_id2sym(inf["vocab"])
    df = inf["model_df"].set_index("trial_id")

    out_path = run_dir / "encoder_predictions.jsonl"
    if split == "val":
        out_path = run_dir / "encoder_predictions_val.jsonl"

    n = 0
    with out_path.open("w") as fh:
        for batch in inf["results"]:
            greedy = greedy_decode(batch["logits"], lengths=batch["out_len"], blank=blank_id)
            if "token" in str(run_dir):
                nbest = nbest_decode(batch["logits"], lengths=batch["out_len"],
                                    blank=blank_id, beam_size=beam_size, n_best=n_best,
                                    token_beam=token_beam)
            for i in range(len(greedy)):
                tid = batch["trial_id"][i]
                gt_ids = batch["labels"][i, : int(batch["label_len"][i].item())].tolist()
                gt_units = ids_to_symbols(gt_ids, id2sym)
                greedy_units = ids_to_symbols(greedy[i], id2sym)
                if "token" in str(run_dir):
                    beam_units = [ids_to_symbols(h, id2sym) for h in nbest[i]]
                    beam_errs = [_seq_err(gt_units, c) for c in beam_units]
                else:
                    beam_units = []
                    beam_errs = None

                gt_text = str(df.loc[tid, "text"]) if tid in df.index else ""
                rec = {
                    "trial_id": tid,
                    "unit": unit,
                    "gt_text": gt_text,
                    "gt_units": " ".join(gt_units),
                    "greedy_units": " ".join(greedy_units),
                    "greedy_err": _seq_err(gt_units, greedy_units),
                    "greedy_ops": align_ops(gt_units, greedy_units),
                    "beam_units": [" ".join(c) for c in beam_units],
                    "beam_errs": beam_errs,
                    "oracle_err": min(beam_errs) if beam_errs else _seq_err(gt_units, greedy_units),
                    "oracle_idx": int(min(range(len(beam_errs)), key=beam_errs.__getitem__)) if beam_errs else -1,
                    "n_gt_units": len(gt_units),
                }
                fh.write(json.dumps(rec) + "\n")
                n += 1
    print(f"  [{run_dir.name}] wrote {n} {unit} predictions -> {out_path.name}")
    return out_path
