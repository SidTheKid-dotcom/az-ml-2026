"""Length-batched inference with unchanged pair text, truncation and row identity.

Tokenize each bounded group once, group similar lengths, then restore original
pair order. This changes floating-point batching, not the trained model or rule;
freeze this runtime before held-out evaluation and official inference.
"""
from __future__ import annotations

from contextlib import nullcontext
import numpy as np

from .cross_encoder_experiment import record_texts


def minibatch_order(lengths, batch_size, length_bucketing=True):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or batch_size < 1 or np.any(lengths <= 0):
        raise ValueError('Positive token lengths and batch size are required')
    order = np.argsort(lengths, kind='stable') if length_bucketing else np.arange(len(lengths))
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


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
    model.eval()
    cuda = args.device.startswith('cuda')
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    amp = torch.autocast(device_type='cuda', dtype=dtype) if cuda else nullcontext()
    # Keep autocast active across batches so cached weight casts can be reused.
    with torch.inference_mode(), amp:
        for positions in minibatch_order(lengths, args.inference_batch_size, buckets):
            rows = [{name: encoded[name][int(i)] for name in columns} for i in positions]
            tokens = tokenizer.pad(rows, padding=True, pad_to_multiple_of=multiple,
                                   return_tensors='pt')
            tokens = {name: value.to(args.device) for name, value in tokens.items()}
            logits = model(**tokens).logits
            if logits.shape != (len(positions), 1):
                raise ValueError('Expected one match logit per input pair')
            output[ids[positions]] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Non-finite reranker output')
    return output
