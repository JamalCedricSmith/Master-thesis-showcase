import os
# Reduce CUDA fragmentation OOMs (must be set before torch initializes CUDA).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import re
from pathlib import Path
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from decode.decode import (greedy_decode, nbest_decode,
                            build_id2sym, ids_to_symbols)
from decode.inference import predict_logits
from decode.evaluation import compute_error_rate
from data.transform import tokens_from_text

# Decode conditions for the frozen-LLM pipeline. Both feed the SAME branch
# template; they differ only in how many ranked candidates are injected:
#   greedy -> 1 candidate  (argmax CTC path)
#   beam   -> N candidates (n-best prefix beam search, most-likely first)
DECODE_MODES = ("greedy", "beam")

class SequenceCorrector:
    """
    Frozen-LLM post-processor for the decode pipeline:
    encoder posterior -> greedy/beam candidates -> this LLM -> word output -> WER.

    The LLM is never fine-tuned. The whole instruction lives in a plain-text
    template with a single ``{candidates}`` placeholder, loaded from prompts/:
    ``token_prompt.txt`` for the intermediate token branch and ``word_prompt.txt`` for the
    word branch. The ranked candidate list (most-likely first, newline-separated)
    is injected at ``{candidates}`` and the result is sent to the frozen model.
    Both decode conditions reuse the SAME branch template; greedy injects one
    candidate, beam injects N.

    This is also the base class for the shared evaluation framework
    (run_system_evaluation + IV/OOV WER).
    """
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct", device="auto"):
        self.device = self._get_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if self.device == "cuda":
            # 8B in bf16 (~15GB) barely fits a 16GB card and OOMs during
            # generate; 4-bit NF4 drops it to ~5GB, leaving room for the KV-cache.
            quant = BitsAndBytesConfig(load_in_4bit=True,
                                       bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant, device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(self.device)
        self.templates = self._load_templates()

    def _get_device(self, device):
        if device != "auto": return device
        if torch.backends.mps.is_available(): return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_templates(self):
        """Load the two branch templates from prompts/ (token = intermediate token branch,
        word = word branch). Each must contain exactly one ``{candidates}``
        placeholder where the ranked candidate list is injected."""
        prompts_dir = Path(__file__).parent / "prompts"
        templates = {}
        for branch, fname in [("token", "token_prompt.txt"), ("word", "word_prompt.txt")]:
            p = prompts_dir / fname
            if p.exists():
                templates[branch] = p.read_text()
            else:
                templates[branch] = "{candidates}"
        return templates

    def _build_prompt(self, candidates, is_token=True, **kwargs):
        """Inject a ranked candidate list into the branch template.

        ``candidates`` is most-likely-first; it is joined newline-separated and
        substituted at ``{candidates}``.
        """
        is_token_branch = is_token
        template = self.templates.get("token" if is_token_branch else "word", "{candidates}")
        ranked = "\n".join(c for c in candidates if c.strip())
        return template.replace("{candidates}", ranked)

    def _prepare_messages(self, candidates, is_token=True, **kwargs):
        # The template is self-contained, so it is the single user turn — no
        # separate system prompt or few-shot examples.
        return [{"role": "user", "content": self._build_prompt(candidates, is_token, **kwargs)}]

    def _clean_text(self, text):
        """Removes/normalizes unicode punctuation that interferes with WER."""
        if not text: return ""
        # Replace curly apostrophes with straight ones so "ain't" is one word
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Remove common quotes (unicode + ASCII double) and underscores
        text = re.sub(r'[\u201c\u201d\u201e\u201f\u00ab\u00bb_"]', "", text)
        return text

    def _postprocess(self, response):
        """Normalise a raw LLM reply into clean lowercase words for WER.

        The templates ask for the transcription only (no preamble), so take the
        first non-empty line and strip stray punctuation/quotes.
        """
        for line in response.strip().splitlines():
            if line.strip():
                return " ".join(self._clean_text(line).lower().split())
        return ""

    def correct(self, candidates, is_token=True):
        """Correct a single utterance from its ranked candidate list.

        ``candidates`` is a list of candidate strings, most-likely first (length
        1 for greedy, N for beam).
        """
        if not candidates or not any(c.strip() for c in candidates):
            return ""
        return self.batch_correct([candidates], is_token=is_token)[0]

    def batch_correct(self, candidate_lists, is_token=True):
        """Correct a batch of utterances. Each item is that utterance's ranked
        candidate list (most-likely first)."""
        if not candidate_lists: return []

        # Skip utterances with no usable candidates to avoid empty prompts.
        valid_indices = [i for i, c in enumerate(candidate_lists)
                         if c and any(s.strip() for s in c)]
        if not valid_indices:
            return [""] * len(candidate_lists)

        batch_messages = [self._prepare_messages(candidate_lists[i], is_token=is_token)
                          for i in valid_indices]

        model_inputs = self.tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        results = [""] * len(candidate_lists)
        for i, output in enumerate(outputs):
            input_len = model_inputs['input_ids'][i].shape[0]
            decoded = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
            results[valid_indices[i]] = self._postprocess(decoded)
        return results

    def _decode_candidates(self, batch, id2sym, blank_id, decode_mode, n_best, beam_size,
                           token_beam=None):
        """logits -> per-utterance ranked candidate list (most-likely first).

        Symbol-agnostic: ids map to intermediate tokens or words via id2sym, so this serves
        both branches. Each candidate is a space-joined symbol string ready to be
        injected at ``{candidates}``.
          greedy -> [single candidate]
          beam   -> [N candidates], best first
        ``token_beam`` caps the per-frame symbol fan-out of the beam (see
        ``decode.decode.ctc_beam_search_nbest_single``); ``None`` = exact.
        """
        logits, out_len = batch["logits"], batch["out_len"]
        if decode_mode == "beam":
            dec = nbest_decode(logits, lengths=out_len, blank=blank_id,
                               beam_size=beam_size, n_best=n_best, token_beam=token_beam)
            return [[" ".join(ids_to_symbols(h, id2sym)) for h in item] for item in dec]
        # greedy
        dec = greedy_decode(logits, lengths=out_len, blank=blank_id)
        return [[" ".join(ids_to_symbols(p, id2sym))] for p in dec]

    def run_system_evaluation(self, run_dir, splits=["test"], max_samples=10,
                              decode_mode="greedy", n_best=5, beam_size=10, token_beam=None):
        """Evaluate the decode->frozen-LLM pipeline and save to system_eval.json.

        decode_mode : "greedy" (1 candidate, CTC argmax) or "beam" (top-``n_best``
                      candidates from prefix beam search). Both feed the same
                      branch template; only the number of injected candidates
                      differs.
        """
        if decode_mode not in DECODE_MODES:
            raise ValueError(f"decode_mode must be one of {DECODE_MODES}, got {decode_mode!r}")
        run_dir = Path(run_dir)
        results = {}
        ban_words = self._load_banlist(run_dir)

        for split in splits:
            inf = predict_logits(run_dir / "model.pt", split=split)
            id2sym = build_id2sym(inf["vocab"])
            blank_id = inf["config"]["training"].get("ctc_blank", 0)
            # token branch translates intermediate tokens->words; word branch corrects words.
            is_token = inf["config"].get("transform", {}).get("token_based", True)
            df = inf["model_df"].set_index("trial_id")

            metrics = {"refs": [], "hyps": [], "samples": [], "iv_e": 0, "iv_n": 0, "oov_e": 0, "oov_n": 0}

            for batch in inf["results"]:
                candidate_lists = self._decode_candidates(
                    batch, id2sym, blank_id, decode_mode, n_best, beam_size, token_beam)

                fixed_texts = self.batch_correct(candidate_lists, is_token=is_token)

                for i, fixed_text in enumerate(fixed_texts):
                    tid = batch["trial_id"][i]
                    ref_text = str(df.loc[tid, "text"]) if tid in df.index else ""
                    ref_text = self._clean_text(ref_text)

                    # The encoder is trained on the *verbalized* form of the
                    # reference (tokens_from_text expands numerals/dates/times,
                    # e.g. "07:28 PM" -> ["seven","twenty","eight","pm"]). The LLM
                    # produces that spoken form too, so WER must score against the
                    # same normalization — not the raw written text. Using a bare
                    # [a-z']+ regex on "07:28 PM" would strip the digits and leave
                    # just ["pm"], which is what blew WER past 300%.
                    ref_words = tokens_from_text(ref_text)
                    hyp_words = re.findall(r"[a-z']+", fixed_text.lower())

                    metrics["refs"].append(ref_words)
                    metrics["hyps"].append(hyp_words)

                    ie, In, oe, on = self._get_word_errors(ref_words, hyp_words, ban_words)
                    metrics["iv_e"] += ie; metrics["iv_n"] += In
                    metrics["oov_e"] += oe; metrics["oov_n"] += on

                    if max_samples is None or len(metrics["samples"]) < max_samples:
                        metrics["samples"].append({"ref": ref_text, "ref_norm": " ".join(ref_words),
                                                   "candidates": candidate_lists[i], "fixed": fixed_text})
                
                if max_samples and len(metrics["samples"]) >= max_samples:
                    break

            wer_total = compute_error_rate(metrics["refs"], metrics["hyps"])
            wer_iv = metrics["iv_e"] / metrics["iv_n"] if metrics["iv_n"] > 0 else 0.0
            wer_oov = metrics["oov_e"] / metrics["oov_n"] if metrics["oov_n"] > 0 else 0.0
            results[split] = {
                "WER_TOTAL": wer_total,   # standard: all S+D+I / total ref words
                "WER_IV": wer_iv,         # (S+D)/N over in-domain ref words; insertions excluded (Total only)
                "WER_OOV": wer_oov,       # (S+D)/N over banlisted (out-of-domain) ref words
                # back-compat aliases for existing notebooks
                "WER": wer_total, "WER_B_IV": wer_iv, "WER_B_OOV": wer_oov,
                "samples": metrics["samples"]
            }
            
        (run_dir / "system_eval.json").write_text(json.dumps(results, indent=2))
        return results

    def _load_banlist(self, run_dir):
        # 1. Check for the local copy in run_dir (copied during training)
        local_banlist = Path(run_dir) / "banlist.txt"
        if local_banlist.exists():
            return {w.lower() for w in re.findall(r"[a-z']+", local_banlist.read_text())}

        # 2. Fallback to config path relative to REPO_ROOT (Path(__file__).resolve().parent.parent)
        cfg_path = Path(run_dir) / "config.yaml"
        if not cfg_path.exists(): return set()
        cfg = yaml.safe_load(cfg_path.read_text())
        bl_path = cfg.get("dataset", {}).get("banlist_path")
        if not bl_path: return set()
        
        repo_root = Path(__file__).resolve().parent.parent
        full_path = repo_root / bl_path
        if not full_path.exists(): return set()
        return {w.lower() for w in re.findall(r"[a-z']+", full_path.read_text())}

    def _get_word_errors(self, ref, hyp, ban_words):
        """Calculates word errors (IV/OOV) using dynamic programming."""
        n, m = len(ref), len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i-1] == hyp[j-1]: dp[i][j] = dp[i-1][j-1]
                else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        iv_e, iv_n, oov_e, oov_n = 0, 0, 0, 0
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                if ref[i-1].lower() in ban_words: oov_n += 1
                else: iv_n += 1
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1: # Substitution
                if ref[i-1].lower() in ban_words: oov_n += 1; oov_e += 1
                else: iv_n += 1; iv_e += 1
                i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1: # Deletion
                if ref[i-1].lower() in ban_words: oov_n += 1; oov_e += 1
                else: iv_n += 1; iv_e += 1
                i -= 1
            else: # Insertion -> no reference word of either class; per spec the
                  # insertion term lives ONLY in Total WER, not in the IV/OOV buckets.
                j -= 1
        return iv_e, iv_n, oov_e, oov_n
