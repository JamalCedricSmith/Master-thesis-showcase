"""CTC decoding utilities (greedy + beam search)."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def build_id2sym(vocab: Sequence[str], *, token_id_offset: int = 1, blank: int = 0, **kwargs) -> dict[int, str]:
    """Create id -> symbol mapping including blank."""
    offset = kwargs.get("token_id_offset", token_id_offset)
    id2sym = {blank: "<blank>"}
    for i, sym in enumerate(vocab):
        id2sym[i + offset] = sym
    return id2sym


def ids_to_symbols(ids: Sequence[int], id2sym: dict[int, str]) -> list[str]:
    return [id2sym.get(int(i), "?") for i in ids]


def greedy_decode(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    blank: int = 0,
) -> list[list[int]]:
    """Greedy CTC decode.

    Expects logits of shape (B, T, V).
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    
    # Argmax over vocab dimension for each timestep.
    pred = torch.argmax(logits, dim=-1)  # (B, T)
    results: list[list[int]] = []
    
    for i in range(pred.size(0)):
        T = int(lengths[i].item()) if lengths is not None else pred.size(1)
        # Collapse repeats
        seq = torch.unique_consecutive(pred[i, :T])
        # Remove blanks
        out = seq[seq != blank].tolist()
        results.append([int(x) for x in out])
    return results


def _logsumexp(*vals: float) -> float:
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _logaddexp(a: float, b: float) -> float:
    """log(exp(a) + exp(b)), stable, for the 2-argument beam-search hot path.

    Every accumulation in the prefix beam combines exactly two log-scores, so a
    specialised 2-arg form avoids the variadic ``_logsumexp``'s ``max`` +
    generator + ``exp`` overhead (and uses ``log1p`` for accuracy).
    """
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def ctc_beam_search_single(
    log_probs: torch.Tensor,
    *,
    beam_size: int = 10,
    blank: int = 0,
    token_beam: int | None = None,
) -> list[int]:
    """Prefix beam search for a single sequence -> best prefix. log_probs: (T, V).

    Thin wrapper over ``ctc_beam_search_nbest_single`` (``n_best=1``) so the
    single-best and n-best paths cannot drift apart.
    """
    return ctc_beam_search_nbest_single(
        log_probs, beam_size=beam_size, n_best=1, blank=blank, token_beam=token_beam
    )[0]


def beam_decode(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    blank: int = 0,
    beam_size: int = 10,
    token_beam: int | None = None,
) -> list[list[int]]:
    """Beam search CTC decode for a batch.

    Expects logits of shape (B, T, V).
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    log_probs = logits.log_softmax(-1)
    results: list[list[int]] = []
    for i in range(log_probs.size(0)):
        T = int(lengths[i].item()) if lengths is not None else log_probs.size(1)
        seq = ctc_beam_search_single(log_probs[i, :T], beam_size=beam_size, blank=blank,
                                     token_beam=token_beam)
        results.append(seq)
    return results


def ctc_beam_search_nbest_single(
    log_probs: torch.Tensor,
    *,
    beam_size: int = 10,
    n_best: int = 5,
    blank: int = 0,
    token_beam: int | None = None,
) -> list[list[int]]:
    """Prefix beam search returning the top-``n_best`` prefixes. log_probs: (T, V).

    Same prefix beam as before, but collapses to the ``n_best`` highest-scoring
    (deduplicated) prefixes, best first. Feeds Option 3 (N-best -> corrector).

    Speed (this is the hot loop, O(T * beam_size * V)):
      * the posterior row is pulled to Python once with ``.tolist()`` — a single
        bulk copy — instead of a per-cell ``log_probs[t, c].item()`` (a tensor
        scalar dispatch) inside the triple loop;
      * the blank transition is applied once per beam instead of once per symbol;
      * scores combine via the 2-argument ``_logaddexp``;
      * ``token_beam`` caps how many non-blank symbols per frame may extend a
        prefix (the top-k by frame posterior). CTC posteriors are peaky, so the
        long tail of the vocabulary never wins a beam; capping it removes the
        dominant ``V`` factor at large ``beam_size`` for a negligible change to
        the n-best list. ``None`` considers every symbol (exact, original
        semantics, but still much faster than before).
    """
    T, V = log_probs.shape
    lp = log_probs.tolist()                         # (T, V) python floats; no per-cell .item()

    # Per-frame non-blank extension candidates (blank is handled separately).
    if token_beam is not None and token_beam < V - 1:
        topk = torch.topk(log_probs, min(token_beam + 1, V), dim=-1).indices.tolist()
        frame_syms = [[c for c in row if c != blank][:token_beam] for row in topk]
    else:
        nonblank = [c for c in range(V) if c != blank]
        frame_syms = [nonblank] * T

    NEG = -math.inf
    # prefix -> [log P(path ends in blank), log P(path ends in last symbol)]
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, NEG]}

    for t in range(T):
        row = lp[t]
        p_blank = row[blank]
        new_beams: dict[tuple[int, ...], list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            # Blank: stay on the same prefix, folding into its blank-ending mass.
            ent = new_beams.get(prefix)
            stay = _logaddexp(pb + p_blank, pnb + p_blank)
            if ent is None:
                new_beams[prefix] = [stay, NEG]
            else:
                ent[0] = _logaddexp(ent[0], stay)

            last = prefix[-1] if prefix else None
            for c in frame_syms[t]:
                p = row[c]
                new_prefix = prefix + (c,)
                if c == last:
                    # Repeat: a blank must separate the two emissions, so the
                    # extension draws only from the blank-ending mass (pb), while
                    # the non-blank mass (pnb) stays on the current prefix.
                    ent = new_beams.get(prefix)
                    if ent is None:
                        new_beams[prefix] = [NEG, pnb + p]
                    else:
                        ent[1] = _logaddexp(ent[1], pnb + p)
                    ext = new_beams.get(new_prefix)
                    if ext is None:
                        new_beams[new_prefix] = [NEG, pb + p]
                    else:
                        ext[1] = _logaddexp(ext[1], pb + p)
                else:
                    ext = new_beams.get(new_prefix)
                    add = _logaddexp(pb + p, pnb + p)
                    if ext is None:
                        new_beams[new_prefix] = [NEG, add]
                    else:
                        ext[1] = _logaddexp(ext[1], add)

        if len(new_beams) > beam_size:
            beams = dict(sorted(new_beams.items(),
                               key=lambda kv: _logaddexp(kv[1][0], kv[1][1]),
                               reverse=True)[:beam_size])
        else:
            beams = new_beams

    ranked = sorted(beams.items(), key=lambda kv: _logaddexp(kv[1][0], kv[1][1]), reverse=True)
    out: list[list[int]] = []
    for prefix, _ in ranked:
        seq = list(prefix)
        if seq not in out:          # collapsed prefixes are already blank-free
            out.append(seq)
        if len(out) >= n_best:
            break
    return out or [[]]


def nbest_decode(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    blank: int = 0,
    beam_size: int = 10,
    n_best: int = 5,
    token_beam: int | None = None,
) -> list[list[list[int]]]:
    """Batched N-best CTC decode. Returns, per item, a ranked list of id-seqs.

    Expects logits of shape (B, T, V). Symbol-agnostic: ids map to tokens or
    words via the caller's id2sym, so it serves both branches. ``token_beam``
    caps the per-frame symbol fan-out (see ``ctc_beam_search_nbest_single``).
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    log_probs = logits.log_softmax(-1)
    results: list[list[list[int]]] = []
    for i in range(log_probs.size(0)):
        T = int(lengths[i].item()) if lengths is not None else log_probs.size(1)
        results.append(
            ctc_beam_search_nbest_single(
                log_probs[i, :T], beam_size=beam_size, n_best=n_best, blank=blank,
                token_beam=token_beam
            )
        )
    return results


def confusion_decode(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    blank: int = 0,
    topk: int = 3,
    min_prob: float = 0.1,
) -> list[list[list[int]]]:
    """Batched confusion-network decode (Option 2).

    Uses the greedy argmax path to segment the utterance — each maximal run of
    the same non-blank symbol is one emitted *slot*, so the slot sequence equals
    the greedy collapse. Within a slot, frame posteriors are averaged and the
    top-``topk`` non-blank symbols whose mean probability clears ``min_prob`` are
    kept, best first. A confident slot collapses to a single id (== greedy), so
    ``greedy`` is exactly the k=1 special case and peaky frames degrade cleanly.

    Returns, per item, a list of slots; each slot is a rank-ordered id list.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    probs = logits.softmax(-1)
    path = torch.argmax(logits, dim=-1)  # (B, T)
    results: list[list[list[int]]] = []

    for i in range(probs.size(0)):
        T = int(lengths[i].item()) if lengths is not None else probs.size(1)
        slots: list[list[int]] = []
        t = 0
        while t < T:
            sym = int(path[i, t].item())
            t0 = t
            while t < T and int(path[i, t].item()) == sym:  # maximal run
                t += 1
            if sym == blank:
                continue                                     # blanks are not slots
            mean = probs[i, t0:t].mean(dim=0)                # (V,) over the run
            order = torch.argsort(mean, descending=True).tolist()
            ranked = [c for c in order if c != blank][:topk]
            # Always keep rank-1 (the greedy symbol); gate the rest by min_prob.
            slot = [ranked[0]] + [c for c in ranked[1:] if float(mean[c]) >= min_prob]
            slots.append(slot)
        results.append(slots)
    return results
