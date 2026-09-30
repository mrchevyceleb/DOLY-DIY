"""LM Studio client — the brain on Moria.

Streams completions, kills reasoning, falls back to the small model
when the big one is unavailable, and reports timing for tuning.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request


class BrainOffline(Exception):
    pass


class Brain:
    def __init__(self, cfg):
        b = cfg["brain"]
        self.url = b["lm_studio_url"].rstrip("/")
        self.model = b["model"]
        self.fallback_model = b.get("fallback_model")
        self.max_tokens = b["max_tokens"]
        self.temperature = b["temperature"]
        self.reasoning_effort = b.get("reasoning_effort")
        self.timeout = b["request_timeout_s"]
        self.using_fallback = False

    # ------------------------------------------------------------------ health
    def healthy(self, timeout=3.0):
        try:
            req = urllib.request.Request(self.url + "/v1/models")
            urllib.request.urlopen(req, timeout=timeout).read()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------- chat
    def chat_stream(self, messages, on_delta=None, fallback=True, **sampling):
        """Yield text deltas. Falls back to the small model on model errors.

        Returns a generator; total text is also fed to on_delta callback.
        fallback=False (background asks) never loads the non-resident model.
        """
        for model in self._model_candidates()[:None if fallback else 1]:
            try:
                # An empty reply is NOT a reason to fall back: the fallback
                # isn't resident on Moria, and loading it took 15 GB (36s)
                # beside a training run. The caller speaks a retry prompt.
                yield from self._stream_once(model, messages, on_delta, **sampling)
                if model != self.model:
                    self.using_fallback = True
                return
            except BrainOffline:
                raise
            except urllib.error.HTTPError as e:
                if (e.code in (400, 404, 422) and model == self.model and self.fallback_model
                        and fallback):
                    print(f"[brain] {model} rejected ({e.code}); trying fallback", file=sys.stderr)
                    continue
                raise BrainOffline(f"HTTP {e.code} from {model}") from e
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError,
                    json.JSONDecodeError, ValueError) as e:
                raise BrainOffline(str(e)) from e

    def chat(self, messages, fallback=True, **sampling):
        parts = []
        for delta in self.chat_stream(messages, fallback=fallback, **sampling):
            parts.append(delta)
        return "".join(parts)

    def _model_candidates(self):
        cands = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            cands.append(self.fallback_model)
        return cands

    def _stream_once(self, model, messages, on_delta, temperature=None, max_tokens=None):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": True,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        body = json.dumps(payload).encode()
        t0 = time.perf_counter()
        ttft = None
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422) and self.reasoning_effort:
                # retry once without the reasoning param (older servers)
                payload.pop("reasoning_effort", None)
                req = urllib.request.Request(
                    self.url + "/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                resp = urllib.request.urlopen(req, timeout=self.timeout)
            else:
                raise

        with resp:
            saw_done = False
            json_errors = 0
            n_out = 0
            finish, extra = None, set()
            try:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    if line == "data: [DONE]":
                        saw_done = True
                        break
                    try:
                        chunk = json.loads(line[5:])
                        json_errors = 0
                    except json.JSONDecodeError:
                        json_errors += 1
                        if json_errors > 5:
                            raise BrainOffline("protocol garbage from server")
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    finish = choices[0].get("finish_reason") or finish
                    raw_delta = choices[0].get("delta") or {}
                    extra |= set(raw_delta) - {"content", "role"}
                    delta = raw_delta.get("content") or ""
                    if not delta:
                        continue
                    n_out += 1
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                        print(f"[brain] model={model} TTFT={ttft:.0f}ms", file=sys.stderr)
                    if on_delta:
                        on_delta(delta)
                    yield delta
                if not saw_done and n_out == 0:
                    # EOF before any output and before [DONE] = dead stream
                    raise BrainOffline("stream ended before completion with no output")
                if n_out == 0:
                    print(f"[brain] model={model} no content: finish={finish} "
                          f"other={sorted(extra)} {time.perf_counter()-t0:.2f}s",
                          file=sys.stderr)
            except BrainOffline:
                raise
            except Exception as e:
                # any transport/protocol/parse failure mid-stream = brain offline
                raise BrainOffline(f"stream failed: {e}") from e


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def iter_sentences(deltas):
    """Buffer streamed deltas, yield complete sentences as they close."""
    buf = ""
    for delta in deltas:
        buf += delta
        while True:
            m = _SENTENCE_END.search(buf)
            if not m:
                break
            sent, buf = buf[: m.end()].strip(), buf[m.end():]
            if sent:
                yield sent
    tail = buf.strip()
    if tail:
        yield tail


def spoken_sentences(deltas, detailed=False):
    """Bound actual playback even if the model ignores the brevity request.

    Close the HTTP stream on reaching the limit so it stops generating too.
    """
    max_sentences, remaining = (6, 120) if detailed else (2, 35)
    try:
        for index, sentence in enumerate(iter_sentences(deltas)):
            words = sentence.split()
            if len(words) > remaining:
                if index == 0:
                    yield " ".join(words[:remaining]).rstrip(",;:!?.") + "."
                break
            yield sentence
            remaining -= len(words)
            if index + 1 >= max_sentences or remaining == 0:
                break
    finally:
        close = getattr(deltas, "close", None)
        if close:
            close()
