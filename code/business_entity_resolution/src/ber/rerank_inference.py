"""Length-batched inference with unchanged pair text, truncation and row identity.

Tokenize each bounded group once, group similar lengths, then restore original
pair order. This changes floating-point batching, not the trained model or rule;
freeze this runtime before held-out evaluation and official inference.
"""
from __future__ import annotations

from copy import copy
import numpy as np

from .cross_encoder_experiment import _amp_context, blend_probabilities, record_texts


def minibatch_order(lengths, batch_size, length_bucketing=True):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or batch_size < 1 or np.any(lengths <= 0):
        raise ValueError('Positive token lengths and batch size are required')
    order = np.argsort(lengths, kind='stable') if length_bucketing else np.arange(len(lengths))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def _pad_numpy(encoded, columns, positions, lengths, tokenizer, multiple, torch):
    """Pad right-aligned BGE tokens without per-pair dictionaries or tokenizer.pad."""
    if (getattr(tokenizer, 'padding_side', None) != 'right'
            or not isinstance(tokenizer.pad_token_id, int)
            or set(columns) - {'input_ids', 'attention_mask', 'token_type_ids'}):
        raise ValueError('Fast padding needs right-aligned integer-pad-token classifier inputs')
    width = int(np.max(lengths[positions]))
    if multiple is not None:
        if type(multiple) is not int or multiple <= 0:
            raise ValueError('Padding multiple must be positive or None')
        width = (width + multiple - 1) // multiple * multiple
    tokens = {}
    for name in columns:
        pad = tokenizer.pad_token_id if name == 'input_ids' else 0
        values = np.full((len(positions), width), pad, dtype=np.int64)
        for row, position in enumerate(positions):
            size = int(lengths[position])
            original = encoded[name][int(position)]
            if len(original) != size:
                raise ValueError('Tokenizer input fields lost pair alignment')
            values[row, :size] = original
        tokens[name] = torch.from_numpy(values)
    return tokens


def score_reranker_batched(model, tokenizer, reference, pool, pairs, selected, args):
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(pairs),):
        raise ValueError('Selected mask does not align with pair rows')
    ids = np.flatnonzero(selected)
    output = np.zeros(len(pairs), dtype=np.float32)
    if not len(ids):
        return output
    import torch

    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    if left.min() < 0 or left.max() >= len(reference) or right.min() < 0 or right.max() >= len(pool):
        raise ValueError('Pair indices do not match the supplied records')
    left_texts, right_texts = record_texts(reference), record_texts(pool)
    encoded = tokenizer([left_texts[int(left[i])] for i in ids],
                        [right_texts[int(right[i])] for i in ids],
                        padding=False, truncation=True, max_length=args.max_length,
                        return_attention_mask=True)
    columns = [name for name in tokenizer.model_input_names if name in encoded]
    if 'input_ids' not in columns or any(len(encoded[name]) != len(ids) for name in columns):
        raise ValueError('Tokenizer lost input pair order')
    lengths = np.fromiter((len(row) for row in encoded['input_ids']), dtype=np.int32)
    buckets = getattr(args, 'length_bucketing', True)
    multiple = getattr(args, 'pad_to_multiple_of', 8)
    defer_transfer = getattr(args, 'defer_cpu_transfer', False)
    fast_padding = getattr(args, 'fast_padding', False)
    pin_memory = getattr(args, 'pin_memory', False)
    if any(type(flag) is not bool for flag in (defer_transfer, fast_padding, pin_memory)):
        raise ValueError('Inference optimization switches must be booleans')
    model.eval()
    amp = _amp_context(torch, args.device, getattr(args, 'amp', 'bf16'))
    deferred_scores, deferred_positions = [], []
    # Keep autocast active across batches so cached weight casts can be reused.
    with torch.inference_mode(), amp:
        for positions in minibatch_order(lengths, args.inference_batch_size, buckets):
            if fast_padding:
                tokens = _pad_numpy(encoded, columns, positions, lengths, tokenizer, multiple, torch)
            else:
                rows = [{name: encoded[name][int(i)] for name in columns} for i in positions]
                tokens = tokenizer.pad(rows, padding=True, pad_to_multiple_of=multiple,
                                       return_tensors='pt')
            tokens = {name: (value.pin_memory().to(args.device, non_blocking=True)
                             if pin_memory and args.device.startswith('cuda') else value.to(args.device))
                      for name, value in tokens.items()}
            logits = model(**tokens).logits
            if logits.shape != (len(positions), 1):
                raise ValueError('Expected one match logit per input pair')
            with _amp_context(torch, args.device, 'fp32'):
                scores = torch.sigmoid(logits[:, 0].float())
                if defer_transfer:
                    deferred_scores.append(scores)
                    deferred_positions.append(positions)
                else:
                    output[ids[positions]] = scores.cpu().numpy()
        if defer_transfer:
            with _amp_context(torch, args.device, 'fp32'):
                output[ids[np.concatenate(deferred_positions)]] = torch.cat(deferred_scores).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Non-finite reranker output')
    return output


def selective_fp32_mask(base_scores, bf16_scores, selected, *, alpha, threshold, band):
    """Choose an FP32 second pass using only BF16 scores and saved tree scores."""
    base = np.asarray(base_scores, dtype=np.float32)
    bf16 = np.asarray(bf16_scores, dtype=np.float32)
    selected = np.asarray(selected, dtype=bool)
    if (base.ndim != 1 or base.shape != bf16.shape or base.shape != selected.shape
            or not np.isfinite(base).all() or not np.isfinite(bf16).all()
            or np.any((base[selected] < 0) | (base[selected] > 1))
            or np.any((bf16[selected] < 0) | (bf16[selected] > 1))):
        raise ValueError('Selective FP32 scoring requires aligned finite tree/CE probabilities')
    if (not 0 < alpha <= 1 or not np.isfinite(threshold) or threshold <= 0
            or not np.isfinite(band) or band < 0):
        raise ValueError('Selective FP32 blend, threshold or band is invalid')
    blended = blend_probabilities(base, bf16, selected, alpha)
    return selected & (np.abs(np.asarray(blended, dtype=np.float64) - threshold) <= band)


def score_reranker_selective_fp32(model, tokenizer, reference, pool, pairs, selected, args,
                                  base_scores, *, alpha, threshold, band, timings=None):
    """BF16 first, then FP32 only for a fixed, label-free blend-threshold band.

    Return (corrected probabilities, FP32 rescore mask, original BF16 scores).
    The caller supplies its already-gated candidate mask and keeps the same
    threshold, pair order, tokenizer, length sorting and pad8 in both passes.
    """
    if getattr(args, 'amp', 'bf16') != 'bf16':
        raise ValueError('Selective FP32 scoring needs a BF16 first pass')
    if np.asarray(base_scores).shape != (len(pairs),) or np.asarray(selected).shape != (len(pairs),):
        raise ValueError('Selective FP32 scoring requires aligned base scores and candidate mask')
    if timings is not None:
        import time
        import torch

        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
    bf16 = score_reranker_batched(model, tokenizer, reference, pool, pairs, selected, args)
    if timings is not None:
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        after_bf16 = time.perf_counter()
    near = selective_fp32_mask(base_scores, bf16, selected, alpha=alpha, threshold=threshold, band=band)
    if timings is not None:
        after_selection = time.perf_counter()
    corrected = bf16.copy()
    if near.any():
        fp_args = copy(args)
        fp_args.amp = 'fp32'
        fp32 = score_reranker_batched(model, tokenizer, reference, pool, pairs, near, fp_args)
        corrected[near] = fp32[near]
    if timings is not None:
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        finished = time.perf_counter()
        timings.update(bf16_seconds=after_bf16-started, selection_seconds=after_selection-after_bf16,
                       fp32_seconds=finished-after_selection, total_seconds=finished-started,
                       bf16_pairs=int(np.asarray(selected, dtype=bool).sum()), fp32_pairs=int(near.sum()))
    return corrected, near, bf16
