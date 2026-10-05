#!/usr/bin/env python3
"""Build a portable audition player from audition_local_tts.py manifests."""
import argparse
import html
import json
import wave
from pathlib import Path
from urllib.parse import quote


def build(folder):
    manifests = [json.loads(path.read_text(encoding="utf-8"))
                 for name in ("piper.json", "kokoro.json", "qwen.json", "qwen-robots.json", "qwen-mixed.json", "qwen-kids.json")
                 if (path := folder / name).exists()]
    cards = []
    for manifest in manifests:
        for voice in manifest["voices"]:
            name = voice["file"]
            path = (folder / name).resolve()
            if path.parent != folder.resolve():
                raise ValueError("audio must be in the audition directory")
            with wave.open(str(path), "rb") as audio:
                if not audio.getnframes():
                    raise ValueError(f"empty audio: {name}")
            measurements = voice["measurements"]
            if measurements:
                times = [item["synthesis_s"] for item in measurements]
                speed = sum(times) / sum(item["audio_s"] for item in measurements)
                kind = html.escape(voice.get("benchmark_kind", "Short-line benchmark"))
                timing = f'{times[0]:.2f}s' if len(times) == 1 else f'{min(times):.2f}–{max(times):.2f}s'
                stats = f'<span>{kind} <strong>{timing}</strong></span><span>Benchmark speed <strong>{1 / speed:.1f}× realtime</strong></span>'
            else:
                stats = '<span>Original take · tuned robot character</span>'
            title = html.escape(voice["title"])
            source = "HTTP synthesis" if manifest["backend"] == "piper" else "Local model"
            cards.append(f'''<article id="{html.escape(voice['id'])}"><div class="eyebrow">{html.escape(manifest['backend'])} · {source}</div>
<h2>{title}</h2><audio controls preload="metadata" aria-label="Listen to {title}" src="{quote(name)}"></audio>
<p class="note">{html.escape(voice.get('description', ''))}</p><div class="stats">{stats}</div>
<a class="download" href="{quote(name)}" download>Download WAV</a></article>''')
    if not cards:
        raise ValueError("no completed auditions found")
    lines = manifests[0]["lines"]
    if any(manifest["lines"] != lines for manifest in manifests):
        raise ValueError("auditions must use the same transcript")
    transcript = " ".join(html.escape(line) for line in lines)
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Spark · Voice auditions</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101419;color:#edf0f5}
*{box-sizing:border-box}body{margin:0}main{max-width:1080px;margin:auto;padding:52px 24px}
.eyebrow{color:#93c7bc;font-size:12px;font-weight:650;letter-spacing:.13em;text-transform:uppercase}
h1{font-size:clamp(32px,5vw,52px);letter-spacing:-.05em;margin:12px 0}h2{font-size:21px;margin:12px 0 24px}
.intro{max-width:690px;color:#bdc4cf;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:32px 0}
article{background:#1a2028;border:1px solid #313b48;border-radius:18px;padding:24px}
audio{width:100%;height:44px}.stats{display:flex;gap:24px;margin:20px 0;color:#bdc4cf;font-size:12px}
strong{display:block;color:#edf0f5;font-size:16px;margin-top:5px}.download{color:#9bd6c9;font-size:13px;text-underline-offset:4px}
.script{background:#161b22;border-left:3px solid #93c7bc;padding:20px 24px;line-height:1.7;color:#d1d7e0}
.note{color:#9da9b9;font-size:13px;line-height:1.7;max-width:850px}a:focus-visible,audio:focus-visible{outline:3px solid #93c7bc;outline-offset:5px}
@media(max-width:650px){main{padding:30px 16px}.grid{grid-template-columns:1fr}.stats{flex-wrap:wrap}}
</style></head><body><main><div class="eyebrow">Spark / voice lab / 05 October 2026</div>
<h1>Find Spark's voice.</h1><p class="intro">Same lines. Different local engines. Listen for warmth, clarity, pacing and personality.
Choose the voice you would enjoy talking to every day. Playing a sample pauses the others.</p>
<div class="grid">CARDS</div><p class="eyebrow">The audition script</p><div class="script">TRANSCRIPT</div>
<p class="note">Timings measure complete clips after warmup; they exclude model loading, speech recognition, the brain and playback.
They are not time to first streamed audio or end-to-end conversation latency. The current Spark sample includes her existing pitch and robot effects;
the three Kokoro voices and the original Warm Companion and Playful Spark are unprocessed. The added robot auditions use gentle pitch and electronic effects.
Each Qwen audition is one continuous take, since separate voice-design calls can change the speaker.
The original Qwen timings use short benchmark lines; new designs report full-audition generation time. A chosen Qwen identity needs fixing before deployment.
These are scripted auditions, so the timer line does not set a real timer.</p>
<p class="note">The voice is one part of the upgrade. Streaming audio, echo cancellation, interruption handling and better turn detection
are the next conversation-layer improvements. Robot actions continue through Spark's existing controller.</p>
</main><script>document.querySelectorAll('audio').forEach(player=>player.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(other=>{if(other!==player)other.pause()});
}));</script></body></html>'''
    target = folder / "index.html"
    target.write_text(page.replace("CARDS", "\n".join(cards)).replace("TRANSCRIPT", transcript), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    build(parser.parse_args().folder)
