"""Acoustic AEC check. Run ONLY with spark-brain stopped; never initializes SDK.

python tools/check_duplex_audio.py --config config.json --wav chosen-voice.wav
Requires numpy, already available on Spark. Prints metrics, stores no audio.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from spark.duplex import EchoCanceller, TurnInterrupted
from spark.ear import MicStream
from spark.streamspeech import play_wav
from spark.voicefx import audioop


def main():
    import subprocess
    if subprocess.run(["systemctl", "is-active", "--quiet", "spark-brain"]).returncode == 0:
        raise SystemExit("Stop spark-brain first; it owns the microphone.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--capture-latency-ms", type=float, default=None)
    ap.add_argument("--cancel-after", type=float)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rate = cfg["audio"]["sample_rate"]
    if args.capture_latency_ms is None:
        args.capture_latency_ms = cfg["audio"].get("capture_latency_ms", 40)
    echo = EchoCanceller(rate, capture_latency_ms=args.capture_latency_ms)
    captures, renders, used = [], [], []
    state = None

    class Recorder:
        def capture(self, pcm, stamp):
            clean = echo.process(pcm, stamp)
            captures.append((stamp, pcm, clean))
            used.append(echo.last_reference)
            return clean

    def render(pcm, hz, when):
        nonlocal state
        echo.render(pcm, hz, when)
        data, state = audioop.ratecv(pcm, 2, 1, hz, rate, state)
        renders.append((when, data))

    cancel = threading.Event()
    try:
        with MicStream(cfg) as mic:
            mic.processor = Recorder()
            time.sleep(.5)
            started = time.monotonic()
            if args.cancel_after:
                threading.Timer(args.cancel_after, cancel.set).start()
            try:
                play_wav(args.wav, {"stream_volume": cfg.get("sounds", {}).get("volume", 90)},
                         lambda until: None, cancel, render)
            except TurnInterrupted:
                print(f"cancel_to_player_exit_ms={(time.monotonic()-started-args.cancel_after)*1000:.1f}")
            time.sleep(.5)
    finally:
        echo.close()
    raw = np.frombuffer(b"".join(p for _, p, _ in captures), dtype="<i2").astype(float)
    clean = np.frombuffer(b"".join(p for _, _, p in captures), dtype="<i2").astype(float)
    reference = np.zeros_like(raw)
    # Map the planned speaker timeline to capture end timestamps. FFT
    # correlation then estimates any remaining physical timing mismatch.
    origin = captures[0][0] - .02 - args.capture_latency_ms/1000
    for when, pcm in renders:
        at = round((when-origin)*rate)
        data = np.frombuffer(pcm, dtype="<i2").astype(float)
        if at >= 0 and at+len(data) <= len(reference):
            reference[at:at+len(data)] = data
    size = 1 << (len(raw)*2-1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(raw, size)*np.conj(np.fft.rfft(reference, size)), size)
    window = int(rate*.5)
    shifts = np.arange(-window, window+1)
    best = int(shifts[np.argmax(np.abs(corr[shifts % size]))])
    start = min(len(raw)//2, int(rate*4))
    stop = len(raw)-int(rate*.5)
    used_ref = np.frombuffer(b"".join(used),dtype="<i2").astype(float)
    corr_used = np.fft.irfft(np.fft.rfft(raw,size)*np.conj(np.fft.rfft(used_ref,size)),size)
    used_lag = int(shifts[np.argmax(np.abs(corr_used[shifts % size]))])
    raw_power = np.mean(raw[start:stop]**2)
    clean_power = np.mean(clean[start:stop]**2)
    print(json.dumps({"capture_interval_min_max_ms": [round(float(x)*1000,1) for x in [np.min(np.diff([t for t,_,_ in captures])), np.max(np.diff([t for t,_,_ in captures]))]], "used_ref_lag_ms": round(used_lag*1000/rate,1), "used_ref_rms": round(float(np.sqrt(np.mean(used_ref**2)))), "aec": "Speex", "duration_s": len(raw)/rate,
                      "reference_to_mic_lag_ms": round(best*1000/rate, 1),
                      "echo_reduction_db": round(10*np.log10(max(1,raw_power)/max(1,clean_power)), 1),
                      "raw_rms": round(np.sqrt(raw_power)), "clean_rms": round(np.sqrt(clean_power)),
                      "mic_clipped_pct": round(float(np.mean(np.abs(raw)>=32760)*100), 3)}))


if __name__ == "__main__":
    main()
