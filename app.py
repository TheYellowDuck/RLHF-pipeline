"""Web chat UI for a trained RLHF policy (Gradio).

    pip install gradio
    python app.py --model checkpoints/ppo
    python app.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model

Opens a browser chat at http://localhost:7860. With Best-of-N the reward model reranks N samples
per turn and the chosen reward is shown under each reply.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlhf.inference import ChatEngine


def main():
    p = argparse.ArgumentParser(description="Web chat UI for a trained RLHF policy")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="local checkpoint or HF id")
    p.add_argument("--reward-model", default=None, help="reward model for Best-of-N reranking")
    p.add_argument("--best-of-n", type=int, default=1)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--share", action="store_true", help="public Gradio link")
    args = p.parse_args()

    import gradio as gr

    engine = ChatEngine(args.model, reward_model=args.reward_model, best_of_n=args.best_of_n,
                        device=args.device, dtype=args.dtype)

    def respond(message, history):
        messages = [{"role": "system", "content": args.system}] if args.system else []
        for h in history:                       # gradio 'messages' format: [{role, content}, ...]
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})
        reply, info = engine.reply(messages, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature)
        if info:
            reply += f"\n\n_(best-of-{info['n']} · reward {info['reward']:.2f} vs mean {info['mean']:.2f})_"
        return reply

    mode = f"Best-of-{args.best_of_n} (reward-model reranked)" if engine.rm else "single-sample"
    gr.ChatInterface(
        respond, type="messages",
        title=f"RLHF chat — {args.model}",
        description=f"Policy trained with this repo's SFT→RM→PPO pipeline. Decoding: {mode}.",
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
