"""Zero-dependency web chat UI for a trained RLHF policy.

No extra packages — Python's standard library plus this repo's deps.

    python app.py --model checkpoints/ppo
    python app.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model

Open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rlhf.inference import ChatEngine

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>RLHF Chat</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0b0d12;--panel:#12151c;--panel2:#171b24;--border:#232936;--text:#e8eaed;--muted:#9aa4b2;
--accent:#3b82f6;--user:#1c3a5e;--code:#0d1117}
*{box-sizing:border-box}
body{margin:0;font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:11px;padding:12px 18px;border-bottom:1px solid var(--border);
background:var(--panel)}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 9px #22c55e}
.title{font-weight:600}.mode{font-size:12px;color:var(--muted);background:var(--panel2);padding:3px 10px;
border-radius:20px;border:1px solid var(--border)}.spacer{flex:1}
header button{background:var(--panel2);border:1px solid var(--border);color:var(--text);padding:6px 12px;
border-radius:8px;cursor:pointer;font-size:13px}header button:hover{background:var(--border)}
#log{flex:1;overflow-y:auto;padding:22px 0}.wrap{max-width:768px;margin:0 auto;padding:0 18px}
.row{display:flex;gap:12px;margin:18px 0;align-items:flex-start}.row.user{flex-direction:row-reverse}
.avatar{width:30px;height:30px;border-radius:8px;flex:none;display:flex;align-items:center;justify-content:center;
font-size:12px;font-weight:600}.user .avatar{background:#2563eb}.bot .avatar{background:#3a4250}
.col{max-width:80%}.user .col{align-items:flex-end;display:flex;flex-direction:column}
.bubble{padding:11px 15px;border-radius:14px;overflow-wrap:anywhere}
.user .bubble{background:var(--user);border:1px solid #2a4a6b;border-top-right-radius:4px}
.bot .bubble{background:var(--panel);border:1px solid var(--border);border-top-left-radius:4px}
.bubble p{margin:0 0 9px}.bubble p:last-child{margin:0}
.bubble pre{background:var(--code);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;margin:9px 0}
.bubble code{background:#ffffff14;padding:2px 5px;border-radius:4px;font-family:ui-monospace,Menlo,monospace;font-size:13px}
.bubble pre code{background:none;padding:0}.bubble ul,.bubble ol{margin:6px 0;padding-left:22px}
.badge{display:inline-block;font-size:11px;color:#a7f3d0;background:#064e3b;border:1px solid #066249;
padding:2px 9px;border-radius:20px;margin-top:7px}
.typing span{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--muted);margin:0 2px;
animation:b 1.2s infinite}.typing span:nth-child(2){animation-delay:.2s}.typing span:nth-child(3){animation-delay:.4s}
@keyframes b{0%,60%,100%{opacity:.3;transform:translateY(0)}30%{opacity:1;transform:translateY(-4px)}}
footer{border-top:1px solid var(--border);background:var(--panel);padding:12px 0}
.inputbar{max-width:768px;margin:0 auto;padding:0 18px;display:flex;gap:10px;align-items:flex-end}
textarea{flex:1;resize:none;background:var(--panel2);border:1px solid var(--border);border-radius:12px;
color:var(--text);padding:12px 14px;font:inherit;max-height:170px;outline:none}
textarea:focus{border-color:var(--accent)}
.send{background:var(--accent);border:0;color:#fff;width:44px;height:44px;border-radius:12px;cursor:pointer;
font-size:18px;flex:none}.send:disabled{opacity:.45;cursor:default}
.hint{max-width:768px;margin:7px auto 0;padding:0 18px;font-size:11px;color:var(--muted)}
.empty{text-align:center;color:var(--muted);margin-top:13vh}.empty h1{font-size:23px;color:var(--text);font-weight:600;margin:0 0 6px}
</style></head><body>
<header><span class="dot"></span><span class="title" id="title">RLHF Chat</span>
<span class="mode" id="mode"></span><span class="spacer"></span>
<button onclick="reset()">＋ New chat</button></header>
<div id="log"><div class="wrap" id="wrap"></div></div>
<footer><div class="inputbar">
<textarea id="m" rows="1" placeholder="Message your model…" autofocus></textarea>
<button class="send" id="send" title="Send">↑</button></div>
<div class="hint">Enter to send · Shift+Enter for newline · runs locally on your machine</div></footer>
<script>
const log=document.getElementById('log'),wrap=document.getElementById('wrap'),ta=document.getElementById('m'),send=document.getElementById('send');
let hist=[],busy=false;
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function md(t){t=esc(t);
 t=t.replace(/```([\s\S]*?)```/g,(m,c)=>'<pre><code>'+c.replace(/^\n/,'')+'</code></pre>');
 t=t.replace(/`([^`\n]+)`/g,'<code>$1</code>');
 t=t.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
 let out=[],ul=false;
 for(const ln of t.split('\n')){
  if(/^\s*[-*]\s+/.test(ln)){if(!ul){out.push('<ul>');ul=true;}out.push('<li>'+ln.replace(/^\s*[-*]\s+/,'')+'</li>');}
  else{if(ul){out.push('</ul>');ul=false;}out.push(ln);}}
 if(ul)out.push('</ul>');
 return out.join('\n').split(/\n{2,}/).map(b=>/^\s*<(pre|ul|ol)/.test(b)?b:'<p>'+b.replace(/\n/g,'<br>')+'</p>').join('');
}
function showEmpty(){wrap.innerHTML='<div class="empty"><h1>Chat with your model</h1><p>Trained with this repo&#39;s SFT &rarr; reward-model &rarr; PPO pipeline.</p></div>';}
function bubble(role,html,meta){const e=wrap.querySelector('.empty');if(e)e.remove();
 const row=document.createElement('div');row.className='row '+role;
 row.innerHTML='<div class="avatar">'+(role==='user'?'You':'AI')+'</div><div class="col"><div class="bubble">'+html+'</div>'+(meta?'<div class="badge">'+esc(meta)+'</div>':'')+'</div>';
 wrap.appendChild(row);log.scrollTop=log.scrollHeight;return row;}
async function submit(){const text=ta.value.trim();if(!text||busy)return;busy=true;send.disabled=true;
 ta.value='';ta.style.height='auto';bubble('user',md(text));hist.push({role:'user',content:text});
 const t=bubble('bot','<div class="typing"><span></span><span></span><span></span></div>');
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:hist})});
  const j=await r.json();t.remove();bubble('bot',md(j.reply),j.info||null);hist.push({role:'assistant',content:j.reply});}
 catch(err){t.remove();bubble('bot','<i>error: '+esc(''+err)+'</i>');}
 busy=false;send.disabled=false;ta.focus();}
function reset(){hist=[];showEmpty();ta.focus();}
send.onclick=submit;
ta.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submit();}});
ta.addEventListener('input',()=>{ta.style.height='auto';ta.style.height=Math.min(170,ta.scrollHeight)+'px';});
fetch('/info').then(r=>r.json()).then(j=>{document.getElementById('title').textContent=j.model.split('/').pop();document.getElementById('mode').textContent=j.mode;});
showEmpty();
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
    Handler.mode = (f"Best-of-{args.best_of_n} · reward-model reranked · {Handler.engine.device}"
                    if Handler.engine.rm else f"single-sample · {Handler.engine.device}")

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  ✅ Open  http://localhost:{args.port}  in your browser   (Ctrl-C to stop)\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
