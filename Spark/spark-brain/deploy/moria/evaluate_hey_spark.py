"""Calibrate on ambient audio, then test the actual streaming wake pipeline.

Uses only held-out speech and the separate openWakeWord validation features.
No microphone or robot SDK is opened. Run after train_hey_spark.py.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import onnxruntime as ort
from train_hey_spark import load_audio


def evaluate(root, runtime):
    assets = root/'assets'
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    head = ort.InferenceSession(str(assets/'hey_spark.onnx'), opts,
                               providers=['CPUExecutionProvider'])
    key = head.get_inputs()[0].name
    background = np.load(assets/'validation.npy', mmap_mode='r')
    # First third is reserved for training by train_hey_spark.py.
    boundary = len(background)//3
    background = background[boundary:]
    scores = []
    for start in range(0, len(background)-15, 512):
        stop = min(start+512, len(background)-15)
        windows = np.stack([background[i:i+16] for i in range(start, stop)])
        scores.extend(head.run(None, {key: windows})[0].reshape(-1))
    scores = np.asarray(scores)
    calibration = scores[:boundary-15]
    evaluation = scores[boundary:]
    calibration_hours = boundary*.08/3600
    hours = (len(background)-boundary)*.08/3600
    required_hits = 2

    def events(threshold, values):
        passing = np.convolve((values >= threshold).astype(int), np.ones(required_hits,dtype=int), 'valid')
        hits = np.flatnonzero(passing == required_hits)+required_hits-1
        count, last = 0, -100
        for i in hits:
            if i-last >= 19:
                count += 1
                last = i
        return count

    # Use ambient validation only to select the lowest safe threshold.
    candidates = [.7, .75, .8, .85, .9, .95, .98, .99, .995, .999]
    threshold = next((t for t in candidates if events(t,calibration)/calibration_hours <= .5), None)
    if threshold is None:
        raise RuntimeError('No threshold meets the ambient false-trigger target: '+
                           str({t:events(t,calibration) for t in candidates}))
    manifest = {'phrases': ['hey spark', 'hey sparks', 'hey sparky'],
                'threshold': threshold, 'consecutive_frames': required_hits,
                'sha256': {n: hashlib.sha256((assets/n).read_bytes()).hexdigest()
                           for n in ('melspectrogram.onnx','embedding_model.onnx','hey_spark.onnx')}}
    (assets/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sys.path.insert(0, str(runtime))
    from neural_wake import NeuralWake
    detector = NeuralWake({'wake': {'words': manifest['phrases'],
                                   'neural_model_dir': str(assets)},
                           'audio': {'sample_rate': 16000}})
    entries = json.loads((root/'audio-manifest.json').read_text())
    results = []
    rng = np.random.default_rng(1709)
    durations = []
    heldout_backgrounds = [e for e in entries if e['split']=='test'
                           and e['kind']=='negative' and e['alias']>=7]
    def score_clip(x):
        x = np.clip(x, -32768, 32767).astype('<i2').tobytes()
        detector.reset()
        detector.last_hit = 0
        run, detected, top = 0, False, 0
        for i in range(0,len(x)-639,640):
            begin=time.perf_counter()
            detector.feed(x[i:i+640])
            if (i//640)%4==3:
                durations.append(time.perf_counter()-begin)
                top=max(top,detector.last_score)
                run=run+1 if detector.last_score>=threshold else 0
                detected |= run>=detector.required_hits
        return bool(detected),float(top)
    for e in entries:
        if e['split'] != 'test':
            continue
        original = load_audio(e['path'])
        for rms in (150, 1200):
            x = original*(rms/max(1, np.sqrt(np.mean(original**2))))
            x = np.pad(x, (16000, 8000))
            x += rng.normal(0, 8, len(x))
            variants=[('quiet',x)]
            if e['label']:
                pool=[b for b in heldout_backgrounds if (b['voice'],b['speaker'])!=(e['voice'],e['speaker'])]
                other=load_audio(pool[int(rng.integers(len(pool)))]['path'])
                other=np.tile(other,1+len(x)//len(other))[:len(x)]
                other*=rms*.5/max(1,np.sqrt(np.mean(other**2)))
                preceding=x.copy();preceding[:16000]+=other[:16000]
                variants.extend((('preceding',preceding),('overlap',x+other)))
            for condition,clip in variants:
                detected,top=score_clip(clip)
                results.append({'text':e['text'],'voice':e['voice'],'speaker':e['speaker'],
                                'rms':rms,'condition':condition,'positive':bool(e['label']),
                                'detected':detected,'peak_score':top})
    positives = [r for r in results if r['positive'] and r['condition']=='quiet']
    mixed = [r for r in results if r['positive'] and r['condition']!='quiet']
    negatives = [r for r in results if not r['positive']]
    report = {'threshold': threshold, 'ambient_hours': hours,
              'calibration_hours': calibration_hours,
              'calibration_false_events': events(threshold,calibration),
              'ambient_false_events': events(threshold,evaluation),
              'ambient_false_per_hour': events(threshold,evaluation)/hours,
              'test_positive_clips': len(positives),
              'test_detected': sum(r['detected'] for r in positives),
              'mixed_positive_clips':len(mixed),
              'mixed_detected':sum(r['detected'] for r in mixed),
              'test_negative_clips': len(negatives),
              'test_false': sum(r['detected'] for r in negatives),
              'feed_p99_ms': float(np.percentile(durations, 99)*1000),
              'misses': [r for r in positives if not r['detected']],
              'false_triggers': [r for r in negatives if r['detected']]}
    (root/'streaming-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('misses','false_triggers')}, indent=2), flush=True)
    if (report['test_detected']/len(positives) < .95
            or report['mixed_detected']/len(mixed) < .95
            or report['test_false']/len(negatives) > .01
            or report['ambient_false_per_hour'] > .5):
        raise RuntimeError('Streaming speech evaluation did not pass the deployment gate')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/home/mrchevyceleb/spark-wake-training'))
    p.add_argument('--runtime', type=Path, required=True, help='Directory containing neural_wake.py')
    a = p.parse_args()
    evaluate(a.root, a.runtime)
