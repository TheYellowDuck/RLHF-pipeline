"""Zero-dependency web chat UI for a trained RLHF policy.

No extra packages — uses only Python's standard library plus this repo's deps.

    python app.py --model checkpoints/ppo
    python app.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlhf.inference import ChatEngine

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>RLHF chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0f1115;color:#e6e6e6}
 h2{font-weight:600} .sub{color:#8a93a3;font-size:13px;margin-top:-10px}
 #log{margin:16px 0;min-height:50vh} .msg{padding:10px 14px;border-radius:12px;margin:8px 0;white-space:pre-wrap}
 .user{background:#1f6feb22;border:1px solid #1f6feb55} .bot{background:#1c1f26;border:1px solid #2b303b}
 .role{font-weight:600;font-size:12px;color:#8a93a3;margin-bottom:2px} .meta{color:#7d8590;font-size:12px}
 form{display:flex;gap:8px;position:sticky;bottom:0;background:#0f1115;padding:10px 0}
 input{flex:1;padding:12px;border-radius:10px;border:1px solid #2b303b;background:#161922;color:#e6e6e6;font-size:16px}
 button{padding:12px 18px;border:0;border-radius:10px;background:#1f6feb;color:#fff;font-weight:600;cursor:pointer}
 button:disabled{opacity:.5}
</style></head><body>
<h2 id="title">RLHF chat</h2><div class="sub" id="sub"></div>
<div id="log"></div>
<form id="f"><input id="m" autocomplete="off" placeholder="Type a message…" autofocus>
<button id="b">Send</button></form>
<script>
const hist=[], log=document.getElementById('log'), inp=document.getElementById('m'), btn=document.getElementById('b');
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function add(role,text,meta){const d=document.createElement('div');d.className='msg '+role;
 d.innerHTML='<div class=role>'+(role==='user'?'you':'bot')+'</div>'+esc(text)+(meta?'<div class=meta>'+esc(meta)+'</div>':'');
 log.appendChild(d);window.scrollTo(0,document.body.scrollHeight);return d;}
fetch('/info').then(r=>r.json()).then(j=>{document.getElementById('title').textContent='RLHF chat — '+j.model;
 document.getElementById('sub').textContent=j.mode;});
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=inp.value.trim();if(!text)return;
 inp.value='';btn.disabled=true;add('user',text);hist.push({role:'user',content:text});
 const t=add('bot','thinking…');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:hist})});
  const j=await r.json();t.remove();add('bot',j.reply,j.info);hist.push({role:'assistant',content:j.reply});}
 catch(err){t.remove();add('bot','[error] '+err);}
 btn.disabled=false;inp.focus();};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    system = None
    gen = None
    title = ""
    mode = ""

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps({"model": self.title, "mode": self.mode}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, "{}")
            return
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + data.get("messages", [])
        reply, info = self.engine.reply(msgs, **self.gen)
        meta = f"best-of-{info['n']} · reward {info['reward']:.2f} (mean {info['mean']:.2f})" if info else None
        self._send(200, json.dumps({"reply": reply, "info": meta}))

    def log_message(self, *a):  # keep the console quiet
        pass


def main():
    p = argparse.ArgumentParser(description="Web chat UI for a trained RLHF policy")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="local checkpoint or HF id")
    p.add_argument("--reward-model", default=None)
    p.add_argument("--best-of-n", type=int, default=1)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    print(f"Loading '{args.model}' (first run downloads ~1 GB; ~20–40 s)…", flush=True)
    Handler.engine = ChatEngine(args.model, reward_model=args.reward_model, best_of_n=args.best_of_n,
                                device=args.device, dtype=args.dtype)
    Handler.system = args.system
    Handler.gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    Handler.title = args.model
    Handler.mode = (f"Best-of-{args.best_of_n} (reward-model reranked) · {Handler.engine.device}"
                    if Handler.engine.rm else f"single-sample · {Handler.engine.device}")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  ✅ Open  http://localhost:{args.port}  in your browser   (Ctrl-C to stop)\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
