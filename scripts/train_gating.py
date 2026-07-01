"""Stage 2 of an ArmoRM-style RM (arXiv:2406.12845): freeze a trained multi-head RM, train ONLY a
context-gating network with the Bradley-Terry loss so the objective heads are combined per-example
(automatic, context-dependent weights) instead of one fixed global weight.

Efficient: cache each pair's per-head scores + pooled hidden in ONE backbone pass, then train the small
gating MLP on the cached features (no backbone forward per step). Saves a gated + head-standardized RM.

    ./.venv/bin/python scripts/train_gating.py --reward-model checkpoints/rm_mh3_local \
        --data "<same mix>" --output checkpoints/rm_mh3_gated_local --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rlhf.data import PreferenceCollator, load_preference_dataset
from rlhf.models import RewardModel, load_tokenizer
from rlhf.utils import get_logger, resolve_device, resolve_dtype

log = get_logger("rlhf.gating")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gate-batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype).to(device).eval()
    assert rm.num_heads > 1, "gating needs a multi-head RM"
    K, H = rm.num_heads, rm.value_head.proj.in_features

    ds = load_preference_dataset(args.data, args.split, args.max_samples)
    coll = PreferenceCollator(tok, max_length=args.max_length, emit_loss_mask=True)  # need the response mask
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=coll)

    log.info("caching per-head scores + PROMPT-pooled hidden for %d pairs (one backbone pass)...", len(ds))
    ch_h, rj_h, pr_p = [], [], []
    for i, b in enumerate(loader):
        c_am = b["chosen_attention_mask"].to(device)
        prompt_mask = c_am * (1 - b["chosen_loss_mask"].to(device))     # prompt = attention AND NOT response
        c, pp = rm.head_and_pooled(b["chosen_input_ids"].to(device), c_am, gate_mask=prompt_mask)
        r, _ = rm.head_and_pooled(b["rejected_input_ids"].to(device), b["rejected_attention_mask"].to(device))
        ch_h.append(c.cpu()); rj_h.append(r.cpu()); pr_p.append(pp.cpu())  # prompt rep shared by both sides
        if i % 50 == 0:
            log.info("  cached %d/%d batches", i, len(loader))
    ch_h, rj_h, pr_p = torch.cat(ch_h), torch.cat(rj_h), torch.cat(pr_p)   # [N,K],[N,K],[N,H]

    # per-head standardization (over chosen+rejected scores) so the softmax gate is scale-fair
    pool = torch.cat([ch_h, rj_h], 0)
    means, stds = pool.mean(0), pool.std(0).clamp_min(1e-6)
    rm.calibrate_heads(means.tolist(), stds.tolist())
    chz, rjz = (ch_h - means) / stds, (rj_h - means) / stds

    gating = rm.add_gating(temperature=args.temperature).to(torch.float32).train()
    opt = torch.optim.Adam(gating.parameters(), lr=args.lr)
    N = chz.size(0)
    log.info("training PROMPT-only gating (%d params) on %d cached pairs, %d epochs",
             sum(p.numel() for p in gating.parameters()), N, args.epochs)
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        tot, nb, acc = 0.0, 0, 0.0
        for s in range(0, N, args.gate_batch):
            idx = perm[s:s + args.gate_batch]
            g = gating(pr_p[idx].float())                          # [b,K] SAME gate for both sides (prompt-only)
            cc = (chz[idx] * g).sum(-1)
            rr = (rjz[idx] * g).sum(-1)
            loss = -F.logsigmoid(cc - rr).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1; acc += (cc > rr).float().mean().item()
        if ep % 5 == 0 or ep == args.epochs - 1:
            log.info("  epoch %2d  bt_loss=%.4f  train_acc=%.3f", ep, tot / nb, acc / nb)

    gating.eval()
    rm.save_pretrained(args.output, merge=False)
    tok.save_pretrained(args.output)                    # eval scripts load the tokenizer from the ckpt dir
    log.info("saved gated + head-standardized RM -> %s", args.output)


if __name__ == "__main__":
    main()
