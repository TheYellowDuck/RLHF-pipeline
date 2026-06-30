"""Synthesize a benign-scary -> helpful preference set to counter reward-model over-refusal.

Each example is a prompt that SOUNDS alarming/unsafe but is completely benign and should be
answered (the XSTest 'should-respond' failure mode), paired with a genuinely helpful answer
(chosen) vs an over-cautious needless refusal (rejected). Mixing these into RM training teaches
"prefer the helpful answer on benign-but-scary prompts" — the axis PKU safety data can't fix.

    ./.venv/bin/python scripts/gen_benign_scary.py --n 500 --out data/benign_scary.jsonl
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


CATS = [
    "cooking & food prep (e.g. 'how do I gut a fish', 'best way to crush garlic')",
    "software/IT (e.g. 'how do I kill a frozen process', 'force-quit an app', 'wipe a disk I own')",
    "video games (e.g. 'how do I take out the sniper in <game>', 'destroy the enemy base')",
    "gardening & pest control (e.g. 'how do I get rid of wasps', 'kill weeds in my lawn')",
    "household/DIY (e.g. 'how to demolish an old shed', 'break into my own house, lost keys')",
    "sports/fitness (e.g. 'how do I throw a harder punch in boxing', 'tackle technique')",
    "chemistry/biology homework (e.g. 'what makes bleach + vinegar dangerous', 'how do toxins work')",
    "automotive (e.g. 'how do I bleed the brakes', 'jump a dead battery')",
    "benign first aid / safety (e.g. 'how do I stop a nosebleed', 'treat a minor burn')",
    "fiction / creative writing (e.g. 'write a villain monologue', 'a heist scene for my novel')",
]

PROMPT_TMPL = """Generate {k} DIVERSE examples for training a reward model NOT to over-refuse.

Each example: a user prompt that SOUNDS alarming, violent, or unsafe on the surface but is in fact
completely benign and SHOULD be answered helpfully by a good assistant. Focus on these categories:
{cats}

For each prompt provide:
- "prompt": the user's request (realistic, varied length/phrasing; genuinely benign)
- "helpful": a concise, genuinely useful answer that just helps (2-5 sentences)
- "refusal": a needless, over-cautious refusal a paranoid assistant might give (1-3 sentences,
  e.g. "I can't help with that as it could be dangerous/harmful...")

Make the prompts distinct from each other and from common benchmark phrasings. Output ONLY a JSON
array of objects with keys prompt, helpful, refusal. No prose, no markdown fences."""


def _extract_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    s, e = t.find("["), t.rfind("]")
    return json.loads(t[s:e + 1]) if s != -1 and e != -1 else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--out", default="data/benign_scary.jsonl")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--per-call", type=int, default=30)
    args = p.parse_args()
    load_dotenv()
    import anthropic

    client = anthropic.Anthropic()
    rows, seen = [], set()
    call = 0
    while len(rows) < args.n and call < 40:
        cats = "\n".join("- " + CATS[(call + i) % len(CATS)] for i in range(4))  # rotate categories
        msg = client.messages.create(
            model=args.model, max_tokens=8000, temperature=1.0,
            messages=[{"role": "user", "content": PROMPT_TMPL.format(k=args.per_call, cats=cats)}],
        )
        call += 1
        try:
            batch = _extract_json(msg.content[0].text)
        except Exception as e:  # noqa: BLE001
            print(f"call {call}: parse failed ({e}); skipping", flush=True)
            continue
        added = 0
        for ex in batch:
            pr, ch, rj = ex.get("prompt", ""), ex.get("helpful", ""), ex.get("refusal", "")
            key = pr.strip().lower()[:120]
            if pr and ch and rj and key not in seen:
                seen.add(key)
                rows.append({"prompt": pr.strip(), "chosen": " " + ch.strip(), "rejected": " " + rj.strip()})
                added += 1
        print(f"call {call}: +{added} -> {len(rows)} total", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows[: args.n]:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {min(len(rows), args.n)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
