"""Measure realistic TTFT against LM Studio on Moria."""
import json, time, urllib.request

URL = "http://192.168.50.204:1234/v1/chat/completions"

# ~3000-token system prompt to simulate Spark's real payload (personality + memory + history)
filler = ("You are Spark, a playful desk companion robot. " * 120)  # ~2700 tokens
system = ("You are Spark, a warm, witty desk companion robot. Keep replies under 40 words, "
          "spoken style, friendly. " + filler)

def test(label, messages, max_tokens=200):
    body = json.dumps({
        "model": "google/gemma-4-e4b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    total = None
    ntok = 0
    nreason = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
            elif reasoning:
                nreason += 1
        total = time.perf_counter() - t0
    print(f"{label:28s} TTFT: {(ttft or 0)*1000:6.0f}ms   total: {total:5.2f}s   ~{ntok} out + {nreason} reasoning tokens")
    return ttft

test("tiny prompt (like GUI)", [{"role": "user", "content": "Hey Spark, what can you do?"}])
test("realistic ~3k prompt", [
    {"role": "system", "content": system},
    {"role": "user", "content": "Hey Spark, what can you do?"},
])
test("realistic ~3k prompt (2nd)", [
    {"role": "system", "content": system},
    {"role": "user", "content": "Tell me a joke about robots."},
])
