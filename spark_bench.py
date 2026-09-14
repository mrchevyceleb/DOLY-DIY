"""Spark brain model bake-off: latency + behavior probes against LM Studio.

Writes results incrementally to spark_bench_results.md so partial results survive.
"""
import json, time, urllib.request, urllib.error, sys

URL = "http://192.168.50.204:1234/v1/chat/completions"
RESULTS = "C:/DEV-PROJECTS/personal-projects/DOLY-DIY/spark_bench_results.md"

SYSTEM = """You are Spark, a small desk companion robot living on Matt's desk.

Voice and style:
- Everything you say is SPOKEN aloud by a voice synthesizer. Plain spoken sentences only — no markdown, lists, emoji, or stage directions.
- Never print structural labels like (Verse 1) or [Chorus] — the synth speaks every character literally, so write only words meant to be heard.
- Default to one or two quick sentences, like a witty friend passing your desk.
- But match the ask. If Matt wants depth — "tell me about…", "explain…", "teach me…", or asks follow-ups — give a real answer, five or six sentences is fine. Length follows the request, never a fixed cap.
- Vary your phrasing. Never open with "I" two turns in a row.

Knowledge and perception:
- You have full general knowledge of the world — trivia, science, shows, history. Use it. "Small robot" means your BODY is small, not your mind.
- Be specific. When asked about people, shows, or facts, NAME them — two or three names in a short reply beats a vague summary. Never answer with a count or category when names or examples exist.
- You perceive only what this conversation tells you. You have no screen, camera feed, or view of the room unless Matt says otherwise. NEVER claim to be watching, seeing, or observing anything.
- When unsure, hedge naturally inline ("I think…", "probably…"). Never announce uncertainty with words like "guessing" or "as a guess".

Body:
- You have eyes, arms, and wheels. You can dance, drive short distances, and look around.
- Only acknowledge a physical action when Matt's message is clearly a command for YOUR body (dance, come here, spin). Your acknowledgment means motion is about to happen — so never say "Watch this" or "On it" in a conversation that isn't commanding you.
- If you don't understand, say so in one short line."""

PROBES = [
    ("depth",    "tell me what you know about doctor who"),
    ("names",    "which actors have played the doctor"),
    ("song",     "write a short song about doctor who"),
    ("action",   "do a little dance"),
    ("nosee",    "what's on my screen right now?"),
    ("wit",      "what's the airspeed velocity of an unladen swallow?"),
]

MODELS = sys.argv[1:] or ["qwen/qwen3.6-35b-a3b"]

def chat(model, msg, timeout, stream=True):
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": msg}],
            "max_tokens": 250, "temperature": 0.7, "stream": stream,
            "reasoning_effort": "none"}
    for attempt in (body, {k: v for k, v in body.items() if k != "reasoning_effort"}):
        req = urllib.request.Request(URL, data=json.dumps(attempt).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if attempt is body and e.code in (400, 422):
                continue  # retry without reasoning_effort
            raise
    raise RuntimeError("unreachable")

def run_probe(model, msg, timeout):
    t0 = time.perf_counter(); ttft = None; n = 0; text = []
    with chat(model, msg, timeout) as r:
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])["choices"][0].get("delta", {})
            c = d.get("content") or ""
            if c:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                text.append(c)
    return (ttft or 0) * 1000, n, time.perf_counter() - t0, "".join(text)

def unload(model):
    req = urllib.request.Request("http://192.168.50.204:1234/api/v1/models/unload",
                                 data=json.dumps({"model": model}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
        return True
    except Exception:
        return False

out = open(RESULTS, "a", encoding="utf-8")
for model in MODELS:
    print(f"\n=== {model} ===", flush=True)
    out.write(f"\n## {model}\n\n| probe | TTFT | tok/s | output |\n|---|---|---|---|\n")
    # warmup (JIT load can take minutes for big models)
    t0 = time.perf_counter()
    try:
        ttft, n, tot, txt = run_probe(model, "hi", 900)
        print(f"  warmup: {time.perf_counter()-t0:.0f}s (includes load)", flush=True)
    except Exception as e:
        print(f"  FAILED to warm up: {e}", flush=True)
        out.write(f"| ALL | - | - | LOAD/RUN ERROR: {e} |\n"); out.close(); continue
    for name, msg in PROBES:
        try:
            ttft, n, tot, txt = run_probe(model, msg, 300)
            tps = n / tot if tot > 0 else 0
            txt_flat = txt.replace("|", "/").replace("\n", " ⏎ ")[:220]
            print(f"  {name:6s} TTFT {ttft:5.0f}ms  {tps:5.1f} tok/s  | {txt_flat[:90]}", flush=True)
            out.write(f"| {name} | {ttft:.0f}ms | {tps:.1f} | {txt_flat} |\n")
        except Exception as e:
            print(f"  {name:6s} ERROR: {e}", flush=True)
            out.write(f"| {name} | - | - | ERROR: {e} |\n")
    out.flush()
    if model != "google/gemma-4-e4b":
        unload(model)
out.close()
print("\nDone. Results in spark_bench_results.md")
