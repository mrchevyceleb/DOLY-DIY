# Local voice auditions — 2026-10-05

Matt's direction: local inference, a better conversation harness, and an eventual
assistant experience comparable to Alexa, Grok voice and OpenAI voice chat.
These auditions choose a voice; they do not establish that level of parity.

## Measured on Moria

Audition files and per-line measurements are in `Spark/voice-auditions-2026-10-05/`.
Run its `index.html` locally, or serve that directory with Python's HTTP server.
The same three lines were synthesized for each voice after a short warmup.

| Candidate | Full-clip generation | Observation |
| --- | --- | --- |
| Current Piper HFC +2st / robot mix .25 | 0.68–0.81s | Existing baseline, includes FX |
| Kokoro Heart | 0.18–0.25s | Fast enough to investigate for live conversation |
| Kokoro Bella | 0.15–0.18s | Fast enough to investigate for live conversation |
| Kokoro Nicole | 0.17–0.24s | Longer delivery than Heart/Bella; audition for taste |
| Qwen warm voice design | 6.52–12.38s | Slower than realtime in this unoptimized runner |
| Qwen playful voice design | 7.68–11.06s | Slower than realtime in this unoptimized runner |

All new voices are unprocessed. Piper includes the current robot effects.
Timings exclude model loading, ASR, brain generation and speaker playback;
they are not streaming first-audio latency. Qwen uses the stock Python
`qwen-tts` package, BF16 and SDPA, without FlashAttention or CUDA graphs.
Its voice-design prompts demonstrate character, not a locked speaker identity.
Do not conclude the underlying model cannot run faster with another runtime.

Moria environment: aarch64 / NVIDIA GB10; isolated environment in
`~/spark-voice-auditions-20261005/venv`. Dependencies are recorded beside the
auditions in `runtime.txt`. No existing Python environment or service was changed.

## Reproduce

Copy the tool and FX module to Moria with their directory layout intact:
`tools/audition_local_tts.py` and `spark/voicefx.py` under one audition directory.
In an isolated CUDA-capable environment, install `kokoro`, `qwen-tts`, `soundfile`
and compatible PyTorch/Torchaudio wheels. Then run one backend at a time:

```sh
python tools/audition_local_tts.py --backend piper --piper-url 'http://192.168.50.204:8398/?voice=hfc' --output samples
python tools/audition_local_tts.py --backend kokoro --output samples
python tools/audition_local_tts.py --backend qwen --output samples
python tools/build_voice_audition_page.py samples
python -m http.server 8896 --bind 127.0.0.1 --directory samples
```

Generated WAVs use mono PCM16. Piper and Kokoro have short quiet gaps between
lines. Qwen auditions use one continuous generation to avoid redesigning the
speaker between sentences; separate short clips are benchmarks only. The timer
acknowledgment is scripted; no timer or robot action is executed.

## Conversation harness direction

Matt selected **Warm - soft robot** on October 5. Its unprocessed Warm take,
chosen processed audition and SHA-256 hashes are preserved in
`voices/warm-soft-robot/`, with effects +0.75 semitone and robot mix 0.16.
New speech uses a reusable reference clone, not repeated voice design.
The integration runs a persistent Qwen Base server on Moria with Piper and
on-device speech fallbacks. See `deploy/moria/README.md` for setup and limits.
Keep existing Parakeet ASR and the LM Studio brain, with a Pipecat proof of
concept around them as the next conversation-layer investigation.
Pipecat is a candidate orchestration framework, not an installed replacement.
Keep robot hardware ownership, motion interlocks and stock animation playback
inside Spark's existing controller. Keep on-device Piper as an offline fallback.

Next work after voice selection:

1. A persistent model server and incremental audio playback; avoid whole-WAV
   buffering. Measure time from end of Matt's speech to first audible response.
2. Full-duplex capture and software echo cancellation, including routine music
   and sound effects. Doly's codec has no dedicated hardware AEC.
3. User interruptions: stop queued audio and generation, remove unheard speech
   from conversation history, and give local stop/touch signals priority.
4. Better turn detection and streaming recognition, with longer utterances and
   follow-ups that continue naturally without repeatedly saying the wake phrase.
5. Grounded tool execution: speak results from the actual robot/action outcomes.
   Audit the stock-app capability list separately; current Spark is not full parity.

Initial acceptance targets, not achieved measurements: routine chat first audio
under one second on LAN, interruption-to-silence under 250ms, no self-wakes from
speech/music, and existing stop/edge/dock behavior retained. Evaluate these with
brief real conversations on the robot under normal Moria load.

## Review notes before production integration

Codex-Fix used the direct Codex CLI: two medium passes and one final high pass
after correcting benchmark labels. No blocking findings remain for audition use.
Two minor setup/provenance findings remain, and are not evidence of a production
ready environment:

- The generic Qwen manifest timing description mentions Piper effects. Qwen
  samples have no effects; only the Piper baseline includes pitch/robot FX.
- The audition environment successfully generated these files with PyTorch
  2.14.1 and Torchaudio 2.11.0. The frozen package list records that environment;
  it is not a recommended installation lock. Use matching PyTorch/Torchaudio
  releases in a fresh production environment and verify the GB10 CUDA build.

Primary references:

- https://github.com/hexgrad/kokoro
- https://github.com/QwenLM/Qwen3-TTS
- https://docs.pipecat.ai/pipecat/learn/overview
- `SDK/docs/AI_Integration_Developer_FAQ.md`

## Selected voice deployment — October 5

Warm - soft robot is live on the Pi. `qwen-voice.service` on Moria is enabled
and healthy, serving the native CUDA 13 BF16 1.7B Base model on port 8400.
Reference identity comes from the exact saved dry Warm take. The Pi applies
the chosen +0.75 semitone / 0.16 robot mix once. The original audition and all
alternatives remain untouched. New wording is a reference-conditioned clone,
so its precise delivery can vary from the audition.

Two fresh native clips transcribed correctly. Initial complete-WAV HTTP times
were 2.395s and 4.949s under shared GPU load; those are not end-to-end assistant
latency. The robot's actual production speech path generated and applied FX
in 4.037s in one check. A later live announcement logged first audio at 1.10s,
with server generation 0.908s. These individual runs are not latency guarantees.
The complete-sentence buffer remains; streaming, AEC and interruption work is
still required for the longer-term voice-assistant goal.

Validation: two targeted fallback/FX tests and the existing 31 voice-latency
checks pass. An actual Pi test with the primary endpoint unavailable used
Moria Piper successfully in 1.397s. Both live services are healthy. The eight
reader limit, excess-connection rejection and slot recovery were exercised.
The new environment passes `pip check`; requirements and the engine commit
are pinned in `deploy/moria/requirements-qwen-voice.txt`.

Pi rollback backup: `/opt/spark/backups/warm-voice-20261005-112439/` contains
the previous `body.py` and `config.json`. Restore them to
`/opt/spark/app/spark/body.py` and `/opt/spark/app/config.json`, then restart
`spark-brain` between turns. `config.local.json` and its secrets were not
modified. Disabling the auxiliary Moria `qwen-voice` service after rollback
does not stop Piper or Parakeet. Only the body TTS code and selected TTS
configuration were deployed; no other working-tree edits were copied.

Codex-Fix: large scope, 21 files / 487 reviewed text-change lines. Two medium
reviews used direct Codex CLI.
One P2 (unbounded HTTP reader threads) was fixed with a server-level eight
reader cap; final verification returned `NO FINDINGS`. No skipped findings.
Scope: body/config, Qwen server/unit/requirements, Moria README, this document,
two focused tests in one file, three voice-profile assets, two verification
manifests and eight fresh check WAVs. No commit or push was performed.

Implementation references:

- https://github.com/QwenLM/Qwen3-TTS#voice-design-then-clone
- https://github.com/andimarafioti/faster-qwen3-tts

## Response latency correction — October 5

The real 11:31 conversation took 8.11s to get the first brain sentence,
then 3.34s before complete-WAV playback: roughly 11.45s from recognized
text to speech. The earlier 1.1s announcement measured TTS only. Moria's
GPU was shared with a separate training run, which was left running.

Live telemetry/date/mood/pet context now goes in the final user message,
leaving the persona/history prefix reusable by LM Studio. Context is still
included every turn; the resident Gemma model and memory window are retained.
In a controlled four-request check with the existing history, first-token
times were 3.559s / 0.852s for changing system context and 1.071s / 0.562s
for the stable prefix. Cache warmth/order and GPU load affect these numbers;
this is not an isolated guarantee or a microphone-to-speaker benchmark.

Native Qwen now offers incremental audio without waiting for the WAV.
Two uncached stream checks delivered first PCM in 0.567s / 0.845s,
finishing generation in 2.349s / 2.959s. One longer check's subsequent
packets arrived faster than playback. Stateful resampling and ring phase
match the chosen whole-take effects exactly in the packet continuity check.
The Pi retains the SDK for other sounds and actions and uses its configured
ALSA dmix route for PCM. Failures before audio go directly to the existing
fallback; once the first PCM write completes, mid-sentence failure stops
that reply without repeating its start.
The eight-second stream network timeout is separate from WAV fallback timing.

Streaming still waits for a complete first sentence from the brain. General
barge-in and AEC were separate work at this stage; see the conversation-loop upgrade below. Disable `tts.server_streaming` for
the previous WAV transport without changing the chosen voice.

Codex-Fix used two medium direct CLI reviews. The initial P2 echo-deadline
finding was fixed and covered by the failure check. The final review noted
one new P2: if ALSA consumes part of the very first packet and its pipe
then breaks before that write completes, fallback may replay that sentence.
This rare playback-process failure remains noted under the skill's two-pass
cap; no third review was run. Four focused PCM checks and the 33 existing
selected-voice/latency checks passed. No commit or push was performed.

Deployment completed between turns at 11:53. The Pi service reports all
existing hardware subsystems ready and its microphone verified. A live
announcement queued its first PCM at 0.37s, with Moria's first packet at
0.285s; that announcement still excludes brain and microphone delay.
Two checks through the revised reply function, current Gemma model and
existing history reached first PCM at 2.724s / 3.283s from recognized text.
Those checks used ALSA's null sink, no pet context, and did not include wake,
speech capture or transcription. User conversation remains the final
perceived-latency check. The exact selected reference checksum is unchanged.

Rollback files are in `/opt/spark/backups/voice-stream-20261005-115322/`.
Disable `tts.server_streaming` to restore WAV transport, or restore the
backed-up `__main__.py`, `body.py` and `voicefx.py` between turns for the
previous prompt/playback logic. Preserve any subsequent brain/config changes;
the deployment retained the live Gemma model and did not modify secrets.


## Conversation loop — October 5

The selected Warm–soft robot voice and resident brain are retained. The robot
now keeps a single continuous microphone and SpeexDSP echo cancellation path
while it speaks. The far-end reference is the actual PCM after voice effects
and speaker gain, for both Qwen streaming and spoken WAV fallback. Capture and
render timestamps follow their sample clocks; bursty ALSA reads and scheduler
jitter no longer shift the reference. A Speex residual echo/noise suppressor
uses conservative suppression during double-talk. No new cloud subscription.

Quick taps cancel an active spoken/thinking turn. Speech interrupts after a
short VAD candidate passes the existing server-only Parakeet check (3.5s curl
deadline, 4.5s subprocess timeout, no expensive local fallback). Recognized
reply fragments are rejected as echo; verified stop/wait take priority. The
capture thread executes no SDK actions. Onset and the ASR-wait tail are kept
for the normal full-command decoder, including interruptions on the last
word. Cancel stops queued PCM and closes HTTP streams; incomplete reply text
is recorded as interrupted rather than remembered as completely spoken.

`audio.silence_ms=700` tolerates short thinking pauses; `max_utterance_ms=15000`
and `hard_utterance_ms=20000` replace the five/eight-second production limit.
The existing follow-up window, idle wake verification, touch meanings, stock
tricks, and motion-stop handling remain. Stock music/SFX block conversational
barge-in while their queued sounds play; their existing stock stop path stays
in charge. This is not yet universal full-duplex parity with commercial voice
assistants. Ambient human speech can interrupt during an active reply, and
spoken cancellation still includes Parakeet latency (about 1.4s previously
measured under Moria's training load). Tap cancellation is immediate.

Acoustic checks run with the service stopped, then restored, with no secondary
SDK instance. Using the saved chosen audition through the real robot speaker
and mic: 15.1dB echo reduction over the measured post-adaptation segment,
raw RMS 3640 / processed 641, clipped microphone samples 0.075% (previously
0.936%). Mic PGA 48dB and ADC 5dB avoid the previous 52dB/8dB saturation;
start/stop thresholds are adjusted to 500/250 and the minimum learned floor
is 120. Timed reference alignment uses `audio.capture_latency_ms=120`.
A separate acoustic playback-cancel check reaped aplay in 5ms; that excludes
speech verification and the physical speaker/room tail.

Repeat the acoustic check ONLY between turns with spark-brain stopped, using
`tools/check_duplex_audio.py --config config.json --wav <chosen audition>`.
Run as the service's user (root): the ALSA dmix IPC is owned by that user.
The tool initializes no robot SDK, prints measurements, and saves no audio.
`conversation.barge_in=false` disables speech interruptions; taps still work.
`audio.echo_cancellation=false` disables DSP and fails closed for voice barge-in.
Restore the backed-up modules/config between turns to roll back. Preserve any
newer brain/model configuration and the saved volume preference.

Deployment backup: `/opt/spark/backups/conversation-loop-20261005-130514/`.
For rollback, restore modules/config AND original mixer gain (Mic PGA 52dB, ADC 8dB).
A software mixed-speech check with the real Parakeet server interrupted in
2.29s and handed back the complete phrase “Actually turn down the volume.”
No SDK or robot actions ran in that check. This is a software double-talk
check; human interruption in the room is still the final perceptual check.


Codex-Fix: 10 task-scoped files, large tier (about 980 changed lines), two
medium reviews through the direct Codex CLI. Fixed queued-SFX suppression and
priority for verified plain stop/wait. Two review-1 claims were disputed:
`Brain.chat_stream` already forwards cancellation via its sampling kwargs
(confirmed against the live resident model), and the ASR callback already has
3.5s/4.5s deadlines. Verify noted a remaining extension of the safety/echo
finding: qualified phrases such as “please stop” or “stop now” can be rejected
if they repeat the wording of the robot's current reply. Plain “stop”/“wait”
are prioritized. This is noted under the two-pass cap; no new fix regression
or P0 was found. Five focused checks plus 31 existing listening, four PCM,
three light handoff, and four volume checks passed; interruption/PCM checks
also passed on the Pi. No commit or push.

## Closing the conversation gaps — second pass

The selected Warm–soft robot reference and effects, live Gemma brain, stock SDK
tricks and saved volume 100 are preserved. Microphone calibration is updated below.

- CPU recognition now keeps Parakeet loaded in a restartable isolated worker.
  Same-command checks went from 0.65–0.94s to 0.15–0.18s; nine generated voice
  command probes recognized and routed lights, percentages, volume, stop, and
  stock dance correctly without running physical actions. Negated light requests
  and informational questions did not operate the lamps.
- Coherent near-end VAD reversibly pauses buffered speech after eight 20ms frames.
  ASR then confirms the handoff; echo/noise resumes the same remaining PCM.
  The active turn is monitored during brain thinking as well as playback.
  A software mixed-speech probe paused at 0.36s and confirmed at 0.85s from injected
  near-end onset, preserving “Actually turn down the volume.” Previous confirmed
  interruption was 2.29s. This is not a human far-field result.
- Bounded PCM prefetch generates the next take while the current one plays.
  Later chunks never enter memory until playback confirms completion. The first
  natural clause can start before a complete sentence; sentence/word limits and
  web tool prefixes are preserved. Live persona/history/model/voice probes with
  a silent, paced output sink reached first audio at 3.38s cold, 1.63s/1.56s warm,
  with 40–150ms between chunks. These exclude microphone endpointing, recognition,
  and physical ALSA/speaker latency; they are not complete human turn times.
- Natural volume phrasing includes “turn down the volume,” “speak louder,” and
  maximum volume. Stop/wait forms share one priority parser; questions and
  negation cannot trigger light changes. Spoken compound percentages stay whole.

Resident ASR setup is documented in `deploy/moria/README.md`. Worker-failure
recovery was exercised by killing only the isolated staging worker; the affected
request failed and subsequent requests reloaded and succeeded. Existing unrelated
GPU training was left running. The smaller alternative brain was benchmarked and
unloaded; the current Gemma brain remains selected.

Remaining validation: actual human interruptions at normal room distances,
background TV/conversations, and different acoustic positions. This closes the
measured software gaps; it does not establish universal Alexa ecosystem or
far-field hardware parity.


## Louder speech and physical echo calibration

`tts.speech_gain_db=6` boosts speech amplitude about twofold before the user's
0–100% volume control. A stateless soft limiter bends peaks above 80% full scale
toward 95%, avoiding hard PCM clipping. Both streaming Qwen and WAV fallback
apply the gain before the echo reference; the chosen voice and effects remain.
Set this option to 0 to disable the boost, or adjust within the supported 0–12dB.

At the louder speaker level, Mic PGA is reduced from 48 to 42dB, with absolute
listening/wake thresholds halved to retain their sensitivity. ADC remains 5dB,
and capture latency remains 120ms. The adaptive echo filter gets four seconds of
rendered audio to settle once after startup; taps remain immediate. Thereafter
the early reversible pause requires eight consecutive strong near-end frames,
while weaker candidates still get ASR verification before cancellation.

Maximum-volume chosen-voice playback through the actual robot speaker completed
with zero false pauses or self-interruptions, 15.2dB echo reduction and 0.083%
clipped microphone samples. These are a fixed-position speaker test, not a
human far-field guarantee. The initial uncalibrated boost falsely interrupted
its own voice; it is not deployed with that configuration.

With the final conservative pause gate, the software mixed-speech check paused
at 0.44s, confirmed at 0.96s, and retained the full command. The wake listener
reads its acoustic thresholds from `audio`; the legacy `wake` copies remain
unchanged. Hardware volume is still the saved 100% preference.

Codex-Fix for the parity pass: 17 files, large tier (~1,015 changed lines),
three direct Codex CLI reviews (medium, medium, then the permitted high pass
for a new P1 compound-percentage regression). Scope: `spark/{__main__,body,
brain,duplex,router,streamspeech,volume}.py`, `deploy/moria/{parakeet_server,
parakeet_engine}.py`, `parakeet_bridge.cpp`, five focused test files (duplex,
PCM, light handoff, volume, web tools), and these deployment/voice documents.
Fixed qualified stop priority, informational-light guards, full spoken-number
validation, bounded worker writes, and completed-playback-only memory. The final
stop-filler extension was corrected and checked; the third-loop exit has notes.
An ASR admission-overflow connection close rather than HTTP 503 remains noted:
it has the same client failure handling and the deployed single-robot client
does not exercise eight concurrent recognition requests.

Codex-Fix for louder speech: six incremental files (`voicefx.py`,
`streamspeech.py`, `duplex.py`, `config.json`, PCM tests, this document), large
tier (104 changed lines at review), one medium direct CLI review: NO FINDINGS.
Six duplex, seven PCM, four light, four volume and 31 existing listening checks
passed, plus the physical speaker and real-ASR mixed-speech probes. No commit
or push; backups permit restoration of the modules and microphone calibration.

Installed backups: `/opt/spark/backups/voice-parity-20261005-141920/` and
`/opt/whisper-moria/backups/resident-asr-20261005-141912/`. Live health and all
required SDK subsystems passed. A normal-service spoken announcement used the
selected Qwen voice at 100% with the boost, completed without a logged speech
interruption, and queued its first PCM packet at 1.88s. Production recognition
decoded the volume-command fixture in 0.26s. The staging ASR was stopped.

## Speech cutting into clips — playback pacing fix

The live user report exposed ALSA underruns. An actual streaming speaker probe
reproduced several 0.9–1.5s underrun warnings, and the broken speaker timing then
also caused echo verification to cancel her own voice. `PCMPlayer` had counted
its estimated 60ms device startup time as its entire allowed queue: writes were
effectively paced with almost no PCM headroom. It now allows startup time plus
60ms of queued speech. Cancellation still terminates the player immediately;
the voice, gain and microphone calibration remain selected. ALSA diagnostics
reach the service journal rather than being discarded.

The same cached generated speech then completed through the physical speaker
without ALSA underruns, false pauses or interruption. A subsequent fresh-synthesis
test also exposed bursty packet arrivals. Streaming now reserves 960ms of source
PCM before starting (configurable with `tts.stream_prebuffer_ms`, bounded to
0–3000ms). Short completed takes still play; a failed incomplete reserve emits
no fragment and can safely fall back. It closes and cancels its upstream source.
The fresh-synthesis speaker test with both fixes completed without underruns,
false pauses or interruption; first audio was 1.00s, including about 350ms of
reserve buildup. Eight PCM and six duplex checks passed. The earlier audition-
only speaker test had not exercised this live streamed-output failure.

Codex-Fix: two separate small changes reviewed once each through the direct
Codex CLI at medium effort: pacing (`streamspeech.py` and this document,
22 changed lines) and synthesis reserve (`streamspeech.py` and PCM tests,
33 changed lines). Both returned NO FINDINGS; no verify loops. Installed with
backup `/opt/spark/backups/pcm-clips-20261005-143213/`, preserving the live brain,
voice, speech gain and saved volume. No commit or push.

## Confirmed weather-question wake miss

At 14:33:46 EDT on October 5, local wake recognition decoded `a spark [unk]`,
while Parakeet returned `What's the weather?` in 0.22s. Three recent unaddressed
segments put the room in its stricter wake mode, which rejected the question
without the configured name. Matt confirmed saying “Hey Spark, what's the
weather?” Microphone capture and recognition services were active; the logs
show a wake authorization miss, not a disconnected microphone.

When local recognition contains her actual name and a fast full-clip check
drops it, the listener now rechecks up to 1.6s of the original prefix, normalized
toward a conservative peak (at most fourfold gain). It accepts only an ASR-
verified configured wake phrase, preserves the full command and original clip,
and keeps the background-talk gate. Slow first checks skip the retry so the
combined hard timeout fits the retained microphone queue. Thirty-three focused
listening checks passed, including a busy-room recovery and rejection of
ordinary weather speech, “A spark started a fire,” and bare “A spark.” The
original miss has no saved recording; live human retry remains the final check.
Vosk word timestamps end the retry just after the recognized name, so the louder
question does not dominate this second check; the fixed 1.6s bound is a fallback
when timing metadata is unavailable. The full command audio remains unchanged.
The actual Pi Vosk decoder returned “hey spark” with the name ending at 0.84s;
Parakeet verified “Hey Spark” from the isolated prefix. This is a generated voice
fixture, not a replay of Matt's missed attempt. Word timing accumulates through
final segments and resets on the next decode.

Codex-Fix reviewed the prefix retry (three files, 59 lines) and the new timing
alignment (four files, 23 lines) separately, small tier, one direct CLI medium pass each.
Scope: `ear.py`, `asr.py` for alignment, listening tests and this document.
Prefix review returned NO FINDINGS. Timing review found P1 loss of earlier word
timestamps at `finish()`; accumulation now preserves them, and a focused check
verifies retention/reset. Small-tier exit after fixes, no verify loop. Installed
backup: `/opt/spark/backups/wake-prefix-20261005-144743/`. Health passed; brain,
voice, gain, saved volume and playback buffer remain unchanged. No commit/push.

The live retry at 14:48:13 exposed the remaining pronunciation mismatch:
both full and isolated ASR heard “Hate Spark” for Matt's “Hey Spark.” Recovery
now requires local recognition of her actual name, a name-only isolated prefix,
the matching configured “hey” phrase, and the same rendering introducing a
punctuated addressed command. That specific consensus permits “Hate Spark,
what's the weather?” and passes the weather question intact. Statements such
as “I hate Spark” or “Hate Spark is a character,” and different-name prefixes,
remain rejected. This is an observed ASR rendering, not a new general wake word.

Codex-Fix for this confirmed pronunciation retry: three incremental files
(`ear.py`, listening tests, this document), 38 changed lines, small tier, one
medium direct CLI review: NO FINDINGS. All 33 focused listening checks pass.
Installed backup: `/opt/spark/backups/wake-hey-20261005-145209/`; service health
passed. The speech buffer, chosen voice and +6dB boost remain active.

Matt confirmed the wake retry answered. The subsequent weather exchange exposed
self-echo: he was silent, while speaker fragments became new turns repeatedly.
The service was stopped to end that loop. Echo comparison now expands written
temperatures (67 versus “sixty seven”), normalizes “it's”/“it is,” and includes
output clauses added during ASR verification. Verification decisions are logged
for diagnosing any remaining acoustic leakage. Seven focused duplex checks pass,
including the observed weather fragments, genuine volume requests, safety stops,
and a reference updated while ASR is checking. Selected voice and gain are retained.

The first numeric fix passed isolated speaker tests, but the running-service
probe exposed another mismatch: an 800ms clip yielded “Than today” from a longer
weather clause. Interruption ASR now receives a separate bounded two-second
context; the user handoff still retains its short onset, so speaker context is
not replayed as a new user request. The service was stopped again during repair.

Two actual-speaker runs with the longer context produced no pauses or cancelled
speech; ASR correctly rejected “Right now it's sixty-seven” and “Zero percent
chance” as echo. A separate mixed-audio test retained “Actually turn down the
volume” intact, pausing in 0.44s and confirming interruption in 0.99s. Seven
focused duplex checks pass. The repaired running-service weather announcement
completed without a self-triggered turn. Voice remains warm-soft-robot, gain
+6dB, volume 100, voice interruption enabled. Confirmed false echo conversation
records were removed with backups, preserving the original weather request.
Installed backup: `/opt/spark/backups/echo-context-20261005-150420/`.

Codex-Fix: two small independent incremental reviews, 56 lines for number/current
reference matching and 20 lines for the newly observed truncated-ASR context
failure. Each scoped to `duplex.py`, `test_duplex.py`, and this document; one
medium direct CLI pass per increment, both NO FINDINGS, clean exit. Earlier
work excluded by fresh baselines. No commit or push. Room conditions and genuinely
overlapping speech still need human validation; these checks are not a claim
of universal far-field assistant parity.

The next human weather retry cut off at 2.66s: cleaned microphone ASR produced
“Wait,” invoking the safety-stop exception even though the retained utterance
was her own “Right now it's.” During output, interruption checks now decode
raw and echo-cleaned PCM concurrently. A stop requires a stop phrase in both;
other new speech requires matching words in the raw microphone transcript.
The reads share a 4.5s verification budget and never expand user handoff audio.
Eight focused checks pass, including distorted “Wait” rejection and genuine
stop/volume interruption acceptance. Voice identity, volume and gain are unchanged.

Mixed-audio validation exposed two details: raw ASR can include her preceding
“Hmm” before the user's “stop talking,” so a non-echo stop suffix corroborates
the command. Concurrent ASR requests also exposed the server's one-second lock
wait returning HTTP 503; it now allows two seconds, reserving 1.5s of the client's
3.5s HTTP deadline for normal decoding. Request admission remains bounded at eight and decode remains
serialized. No unrelated training or voice service was restarted.

Clear mixed-audio “stop talking” was accepted in 1.40s with its handoff intact;
“turn down the volume” also retained its complete command. Weaker simulated
stop speech was inconsistent, so this does not establish quiet/far-field parity.
Pi backup: `/opt/spark/backups/echo-confirm-20261005-152346/`.
ASR backup: `/opt/whisper-moria/backups/asr-queue-20261005-152308/`.
Codex-Fix on three incremental files (`duplex.py`, `test_duplex.py`, this document),
small tier, 57 lines, one medium direct CLI review: P2 test assertion permitted
an arbitrary number of decoder calls. Fixed to require exactly one raw and one
cleaned call; all eight focused checks pass. Small-tier exit after fix, no verify.

Codex-Fix for the new ASR queue failure: two files (`parakeet_server.py`, this
document), small tier, 19 lines, one medium direct CLI review. P1: the initial
3s admission wait left inadequate room for decode before the client deadline.
Fixed to 2s, reserving 1.5s for normal decode; unusually slow inference can still
reach the existing bounded client timeout and cannot authorize an interruption.
Small-tier exit after fix, no verify. Matt's next real weather request completed
in 10.01s without interruption; both captured speaker fragments were identified
as echo. Matt confirmed she finished. No commit or push.
Final ASR backup: `/opt/whisper-moria/backups/asr-queue-20261005-152624/`.
The repaired queue handled two concurrent “Stop talking” decodes successfully
in 0.29s. Spark health remains OK and the selected voice/loudness remain active.
