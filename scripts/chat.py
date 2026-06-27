"""Chat with a trained policy — interactive CLI with optional Best-of-N reranking.

Talk to any local checkpoint (the PPO/SFT policy) or any HuggingFace model id. With
``--best-of-n N --reward-model <path>`` the model samples N replies and returns the one the
reward model scores highest — the inference-time alignment trick that reliably beats a single
sample at ~zero training cost.

    python scripts/chat.py --model checkpoints/ppo
    python scripts/chat.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model
    python scripts/chat.py --model Qwen/Qwen2.5-0.5B-Instruct --once "Explain RLHF in one sentence."

Commands inside the chat: /reset (clear history), /quit (exit).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.inference import ChatEngine


def main():
    p = argparse.ArgumentParser(description="Chat with a trained RLHF policy")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="local checkpoint or HF id")
    p.add_argument("--reward-model", default=None, help="reward model for Best-of-N reranking")
    p.add_argument("--best-of-n", type=int, default=1, help="sample N replies, keep the RM's top pick")
    p.add_argument("--system", default=None, help="optional system prompt")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--greedy", action="store_true", help="deterministic decoding (ignored if best-of-n>1)")
    p.add_argument("--max-length", type=int, default=1024, help="truncation for RM scoring")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--once", default=None, help="send one message, print the reply, exit (non-interactive)")
    args = p.parse_args()

    print(f"Loading '{args.model}' (first run downloads ~1 GB; ~20–40 s)…", flush=True)
    engine = ChatEngine(args.model, reward_model=args.reward_model, best_of_n=args.best_of_n,
                        device=args.device, dtype=args.dtype)
    gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
               top_p=args.top_p, greedy=args.greedy, max_length=args.max_length)
    base = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(base)

    def turn(user_text):
        messages.append({"role": "user", "content": user_text})
        reply, info = engine.reply(messages, **gen)
        messages.append({"role": "assistant", "content": reply})
        print(f"\nbot> {reply}")
        if info:
            print(f"   [best-of-{info['n']}: picked reward {info['reward']:.2f} (batch mean {info['mean']:.2f})]")

    if args.once is not None:
        turn(args.once)
        return

    mode = f"best-of-{args.best_of_n} (RM rerank)" if engine.rm else "single-sample"
    slow = str(engine.device) == "cpu"
    print(f"Chatting with `{args.model}` on {engine.device} — {mode}. /reset to clear, /quit to exit."
          + ("\n(on CPU each reply takes ~20–40 s — it's working, not frozen.)" if slow else ""))
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            messages = list(base); print("(history cleared)"); continue
        turn(user)


if __name__ == "__main__":
    main()
