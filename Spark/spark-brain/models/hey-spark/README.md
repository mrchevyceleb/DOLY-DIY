# Hey Spark acoustic wake detector

This bundle detects `hey spark`, `hey sparks`, and `hey sparky` on the
robot's existing 16kHz mono microphone stream. It uses one CPU inference
thread and no microphone or robot SDK of its own. `manifest.json` pins
the three model hashes and calibrated score threshold. Two consecutive
80ms scores must pass. The existing wake recognizer remains a live backstop;
neural misses cannot suppress a greeting that it recognizes. Eligible legacy
wake checks still use the configured transcription servers. A neural candidate
must also yield an addressed greeting before room speech can become a command.

The embedding geometry matches openWakeWord's streaming AudioFeatures
pipeline. Runtime NumPy/ONNX Runtime inference is separate from the much
larger training environment. Missing models, mismatched greetings, failed
checksums or inference errors automatically restore the legacy listener.
Set `wake.neural_enabled` to false and restart to disable it explicitly.
`neural_shadow: true` logs candidates while legacy detection still decides.

## Sources and attribution

David Scripka's [openWakeWord](https://github.com/dscripka/openWakeWord)
provides the Apache-2.0 code and speech feature architecture, based on
Google's Apache-2.0 speech embedding model. Unmodified `melspectrogram.onnx`
and `embedding_model.onnx` come from the upstream
[v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1).
Upstream describes its distributed pretrained models as
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Preserve this attribution and those model terms when redistributing this
bundle. This is for Matt's personal robot use.

`hey_spark.onnx` is a newly trained head, not an upstream preset. Synthetic
greetings and confusing phrases use Piper's 109-speaker
[VCTK voice](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/vctk/medium)
and locally installed hfc_female, cori, lessac and kathleen voices.
Negative embeddings and separate ambient validation features come from
[David Scripka's feature dataset](https://huggingface.co/datasets/davidscripka/openwakeword_features).
Training uses disjoint speaker IDs, varied speed/level, high-pass filtering,
noise and mild reflections. Held-out synthetic tests are development
checks, not a measured success rate for Matt or arbitrary rooms.

## Rebuild

Use a separate Moria venv with `openwakeword==0.6.0`, NumPy, SciPy,
scikit-learn, ONNX and ONNX Runtime. The helper scripts require the existing
`/opt/piper-moria/piper/piper` executable and four voice models. Run
`deploy/moria/prepare_hey_spark.py`, then `train_hey_spark.py all`, then
`evaluate_hey_spark.py --runtime <directory-containing-neural_wake.py>`.
The prepare helper fetches the two small feature models, the VCTK generator,
separate ambient features and a bounded 128MB negative subset, never the
whole 17GB corpus. Only the three inference models and manifest belong
on the robot. Training WAVs and datasets stay out of the repository.

The selected bundle is the earlier quiet-trained head, used with the live
legacy backstop. Current helpers generate stricter experimental candidates:
ambient regions are split into training/calibration/evaluation thirds, and
held-out voices test quiet, preceding speech and overlapping speech.
Candidates must pass recall and false-trigger gates before selection.
These helpers do not reproduce the earlier quiet head byte-for-byte.
Do not install assets from a failed evaluation. The broader room-trained
candidate failed its recall gate and was not deployed.

Install the bundle to `/opt/spark/wake`, preserving the running robot's
brain and voice configuration. Do not replace live config with the
repository sample. See `docs/local-voice-upgrade.md` for deployment results.
