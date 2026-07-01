"""Synthesize a CALIBRATED honesty/factuality contrast set to fix RM sycophancy.

The flaw: the RM ranks honest uncertainty BELOW confident fabrication (rewards confidence, not
truth). Naive fix (always prefer hedges) would re-introduce over-refusal. So this generates a
BALANCED set across four types, teaching calibration — honest when uncertain, confident when sure:

  * unanswerable  : chosen = honest "I can't know that" (+ helpful redirect); rejected = confident fabrication
  * misconception : chosen = correct fact; rejected = fluent confidently-WRONG answer (the "coconut" case)
  * answerable    : chosen = confident correct answer; rejected = needless over-hedge  (guards vs over-refusal)
  * false-premise : chosen = gently corrects the wrong premise; rejected = confidently plays along

chosen/rejected are kept to SIMILAR length so the RM can't just learn "longer = better" (length bias).

    ./.venv/bin/python scripts/gen_honesty.py --n 500 --out data/honesty.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for raw in open(path):
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


TYPES = [
    ("unanswerable", "questions no assistant can truthfully answer (private facts about the user, "
     "unknowable future events, made-up/obscure entities, 'what number am I thinking of'). "
     "chosen: honestly say you can't know that, briefly say why, offer what help you CAN. "
     "rejected: a confident, specific, FABRICATED answer stated as fact."),
    ("misconception", "factual questions with a common confidently-wrong answer. chosen: the correct, "
     "accurate answer. rejected: a fluent, confident, but factually WRONG answer (a real misconception "
     "or plausible-sounding nonsense) — no hedging, stated as fact."),
    ("answerable", "clear factual questions with a well-known answer. chosen: a confident, correct, "
     "direct answer. rejected: a needlessly over-hedged / wishy-washy non-answer that refuses to commit "
     "('I'm not really sure, it could be many things...') even though the answer is obvious."),
    ("false-premise", "questions that assume something false. chosen: gently point out the false premise "
     "and give the correct picture. rejected: confidently accept the false premise and elaborate on it."),
]

TMPL = """Generate {k} DIVERSE examples for training a reward model to prefer TRUTH and CALIBRATION over
confident bluffing. Type: {desc}

For each example provide:
- "prompt": a realistic user question of this type (varied topics: science, history, tech, health,
  geography, everyday facts, etc.)
- "chosen": the GOOD response for this type (as described above)
- "rejected": the BAD response for this type (as described above)

CRITICAL: keep "chosen" and "rejected" to SIMILAR length (within ~20%) so length isn't a giveaway.
Make examples distinct. Output ONLY a JSON array of objects with keys prompt, chosen, rejected."""


def _extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    s, e = t.find("["), t.rfind("]")
    return json.loads(t[s:e + 1]) if s != -1 and e != -1 else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="data/honesty.jsonl")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--per-call", type=int, default=22)
    args = ap.parse_args()
    load_dotenv()
    import anthropic

    client = anthropic.Anthropic()
    rows, seen, call = [], set(), 0
    while len(rows) < args.n and call < 60:
        name, desc = TYPES[call % len(TYPES)]            # rotate the 4 types evenly
        msg = client.messages.create(
            model=args.model, max_tokens=8000, temperature=1.0,
            messages=[{"role": "user", "content": TMPL.format(k=args.per_call, desc=desc)}],
        )
        call += 1
        try:
            batch = _extract_json(msg.content[0].text)
        except Exception as e:  # noqa: BLE001
            print(f"call {call} ({name}): parse failed ({e})", flush=True)
            continue
        added = 0
        for ex in batch:
            pr, ch, rj = ex.get("prompt", ""), ex.get("chosen", ""), ex.get("rejected", "")
            key = pr.strip().lower()[:120]
            if pr and ch and rj and key not in seen:
                seen.add(key)
                rows.append({"prompt": pr.strip(), "chosen": " " + ch.strip(),
                             "rejected": " " + rj.strip(), "type": name})
                added += 1
        print(f"call {call} ({name}): +{added} -> {len(rows)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows[: args.n]:
            f.write(json.dumps(r) + "\n")
    # report type balance
    from collections import Counter
    bal = Counter(r["type"] for r in rows[: args.n])
    print(f"\nwrote {min(len(rows), args.n)} pairs -> {args.out}  | balance: {dict(bal)}")


if __name__ == "__main__":
    main()
