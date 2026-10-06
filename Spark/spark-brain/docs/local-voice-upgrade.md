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

## Quiet speech within three feet

Matt reported that Spark still needed a loud voice within three feet. With
`audio.adaptive_sensitivity=true`, wake and command energy gates now follow the
measured ambient floor instead of imposing the previous loudness minimums.
The lower floor clamp is 40 RMS; start is at least 90 RMS and 1.8 times the
floor. WebRTC VAD receives at most four times gain for quiet frames, while
captured PCM remains unchanged. Wake authorization still requires her name
and the configured “Hey”; the existing echo and raw-microphone cross-checks
remain enabled.

Remote ASR receives bounded per-clip gain (at most four times, targeting 1000
RMS with peak headroom). Adaptive commands bypass the old 600-RMS silence trim,
which could remove quiet words beside a louder word. Gain does not improve
signal-to-noise ratio or change microphone hardware gain.
Wake checks opt into this gain; interruption confirmation does not, preserving
the raw/clean comparison. A steady 200ms level can recover an underestimated
command noise floor before committing an onset.

A real Vosk/WebRTC VAD/Parakeet probe on the Pi using attenuated synthesized
“Hey Spark. What is the weather?” at 100 RMS missed the wake with the old
settings and recognized it with adaptive sensitivity. This is a software
fixture result, not a measured distance or confirmation of Alexa parity.

Matt's first normal-volume retry still missed. The log captured
“He sparked what's the weather?” at peak 199 RMS, floor 40, with 240ms of
voiced frames. This was a pronunciation rejection rather than absent capture.
An opening “he spark/he sparked” now needs an addressed query and a second,
bounded check of the first 900ms confirming the same name sounds or an exact
“Hey Spark”. The full query must end with a question mark in the transcript.
Bare “he sparked” and narrative “he sparked a discussion”
remain rejected. Slow/unavailable confirmation fails closed.

Codex-Fix: initial quiet-speech patch, six files, small tier (91 changed lines),
one medium direct CLI review. Fixed P1 stale-floor recovery and unintended
gain on interruption verification, plus P2 concurrent-check test timing.
`__main__.py` was added to the fix scope for explicit idle-wake gain opt-in.
The new failed-retry pronunciation patch had a separate small review of three
files: fixed P1 narrative query false wakes with interrogative punctuation,
and P2 prefix identity by requiring “Hey Spark” specifically. No verify rounds
on either small patch. Existing protections remain active; 84 checks pass.

Matt's next retry missed first (“East Park, what's the weather?”, peak 405 RMS)
and worked second (“Hey Spark”, peak 647 RMS), finishing in 9.39s. This still
does not establish reliable normal-volume hearing. Adaptive sensitivity now
also bypasses Speex residual suppression when no speaker reference or aligned
echo tail is active. The adaptive filter continues to update; suppression
stays enabled during output and its tail. Idle microphone audio reaches VAD
and ASR intact, so quiet consonants are not subjected to speaker-echo removal.

The idle bypass was reverted after Matt missed both attempts: the raw noise
floor rose to 227 RMS and the first clip had only 100ms admitted by VAD.
The reviewed bypass patch had no findings, but field evidence rejected it.
The “he sparked” recovery was also removed in favor of independent model
verification rather than accumulating pronunciation aliases.

`spark-wake-verifier.service` runs whisper.cpp base.en on Moria's CPU, port
8397, alongside Parakeet on 8399. No GPU or prompt bias. When local keyword
spotting suggests her name or Parakeet hears a known nearby opening, the
idle wake wrapper asks this second recognizer to confirm the exact configured
wake phrase. Ordinary conversation is not sent for a second opinion. Slow
primary recognition skips the extra check; both checks fit the existing
request budget, reserving 1s for subprocess overhead. Failed or skipped
secondary confirmation returns an explicit nonempty rejection: an empty
result would let an exact local keyword stand. Voice interruption keeps its
existing two Parakeet raw/clean checks and never uses this extra model.

CPU sample checks at 100 RMS: “Hey Spark, what is the weather?” was exact in
0.71s; ordinary weather question, “he sparked a discussion” and “the park is
open today” stayed unchanged in 0.58–0.68s. These are synthesized fixtures.
Soft adaptive “Hey” sessions now require three voiced frames rather than five
to request remote recognition; that lowers the check threshold, not wake
authorization. Armed clips previously fell short of the five-frame gate.

Matt also reported a playback glitch during the next answer. ALSA logged
3269ms, 2498ms and 920ms underruns while two purported interruptions
(“Clear Scott”, “Eyes. Hi and see”) failed raw microphone corroboration.
`conversation.barge_in_early_pause=false` keeps playing until ASR confirms an
actual interruption; tap stop remains immediate. This avoids starving ALSA
while an unconfirmed candidate is decoded. Verified voice interruption still
cancels the player, with the existing recognition delay.

Final field check: Matt confirmed both normal-volume requests within three
feet were heard and both answers were smooth. Logs show complete weather
answers in 9.31s and 7.46s, with speaker fragments classified as echo and no
ALSA underrun. The second wake had peak 273 RMS. “Okay, thanks” closed each
follow-up window correctly. A separate mixed-audio real-ASR probe confirmed
“Stop talking” in 1.19s, without an early pause, and retained the complete
stop command. This is not a quiet-over-speaker or far-field parity claim.

Codex-Fix for the independent verifier: seven files (`asr.py`, `ear.py`,
`__main__.py`, `config.json`, `test_voice_latency.py`, this document,
`spark-wake-verifier.service`), large tier (153 lines at verification), two
medium direct CLI reviews. Fixed P1 rejection handling with a nonempty veto
and P1 timeout overhead reservation. Limited success at the verify stop:
the new P1 requesting a second opinion even for a primary exact wake was
disputed because the second model is deliberately for unclear candidates;
primary exact configured names remain authoritative. New P2 noted, not fixed:
nearby “he sparked” may cause an extra check on narrative speech. That check
cannot authorize the homophone: Whisper must hear the exact configured wake,
and its synthesized narrative negative stayed unchanged.

Codex-Fix for cutouts: four files (`duplex.py`, `test_duplex.py`, `config.json`,
this document), small tier, 16 lines, one medium direct CLI review, no findings.
All 83 targeted checks pass. Nine unique files changed in the final worktree;
failed pronunciation-alias and idle-bypass experiments were removed. The
current backup is `/opt/spark/backups/quiet-listening-20261005-163725/`.
Spark, Parakeet, Qwen and the enabled CPU wake verifier are healthy. The live
Gemma brain, Warm soft robot voice, +6dB speech gain, volume 100, stock actions
and configuration overlay remain intact. No commit or push for this task.

To reproduce the optional verifier on Moria with the existing whisper.cpp
binary and `models/ggml-base.en.bin`: install `deploy/moria/spark-wake-verifier.service`
to `/etc/systemd/system/`, run `systemctl daemon-reload` and
`systemctl enable --now spark-wake-verifier`. Its port is separate from Parakeet.

## Finalized local greeting with agreeing name/command evidence

The next missed request was confirmed by Matt as “Hey Spark, come here”.
Local KWS heard “hey spark”; Parakeet heard “Face Park, come here” and CPU
Whisper heard “Thanks, Park. Come here”. Exact phrase matching vetoed this.
The exact-wake callback now retains the local greeting as evidence. A
finalized, configured “Hey Spark” plus two independently recognized vocative
Spark/Park commands with identical word tokens can recover the dropped s.
Only a short greeting/noise prefix is tolerated; “Central Park” and “park the
car”, missing/contradictory evidence and bare local “Spark” stay rejected.
Partial agreement defers until a local final rather than authorizing early.
The agreed command is preserved verbatim; no robot action runs in recognition.

The partial confirmation is cached with that listening attempt's PCM prefix
and exact local greeting. A matching final releases it without decoding the
same phrase twice; a new segment clears it. The observed local tokens must be
the configured two-word greeting followed only by `[unk]` artifacts. Extra
words cannot qualify as an exact greeting.

Codex-Fix for this follow-up: four files (`ear.py`, `__main__.py`,
`test_voice_latency.py`, this document), small tier (86 changed lines when
reviewed), one medium direct CLI review. Fixed P1 discarded partial evidence
and P2 insufficient validation of observed local tokens. No second review per
the small-tier rule. All 84 targeted checks pass, including the actual
Face/Thanks Park case, rejection cases and a final that cannot decode twice.
This is recognition/state verification, not a new physical-distance result.

The corrected build is live and healthy, with backup
`/opt/spark/backups/quiet-listening-20261005-165526/`. Awaiting another
normal-volume field check; the earlier two successes did not establish
consistent wake reliability. No commit or push for this follow-up.

## Response latency after a long idle period

The 18:32 availability request used the already loaded Gemma model. Brain
first-content time was 4.557s; first playable voice packet took another 2.20s.
LM Studio re-evaluated 1,557 tokens in 2.195s. Its GPU also serves an unrelated
training job; that job was left running. The model has no idle TTL now.

The prompt history can keep up to four extra exchanges, then prune four at
once, retaining the original twelve recent exchanges. This preserves the
cached prefix between pruning points; saved history remains twelve turns.
Extra history is dropped if it exceeds 6,000 characters. Clearing memory
clears both windows. A real-server, text-only replay with the actual persona
and saved history measured later-turn mean TTFT 1.082s with the old sliding
window and 0.617s with a stable prefix (three samples each, shared busy GPU).
This establishes a warm-turn improvement, not a long-idle guarantee.

Exact availability questions now answer locally: “I'm here, Matt. I'm
listening.” That selected-voice take is generated at voice-server startup
and pinned separately from its ordinary LRU cache. It reports current
presence, not overall hardware/model health. Pet/game routing remains first;
questions with additional instructions do not match the fast route.
The 960ms synthesis buffer, speaker pacing and interruption rules are intact.

Installed and healthy: Pi backup `response-latency-20261005-184636`; Moria
voice source backup `qwen_voice_server.py.backup-20261005-184637`. The physical
speaker probe queued the pinned availability reply's first PCM in 0.10s,
without an underrun or self-interruption. This announcement probe exercises
playback, not the microphone/intent route. Normal voiced checks are pending.
Codex-Fix: six files, large tier (115 lines), two medium direct CLI reviews;
fixed P2 orphan assistant at long-history trimming, verification clean.
All 86 targeted checks pass. No commit or push.

Memory paging was also observed: the LM Studio worker had about 1.9GB in
swap, and the previous Qwen service peaked at 844MB swapped. Its freshly
restarted replacement had already accumulated 232MB. The voice unit now has
persistent `MemorySwapMax=0`; the current dedicated LM Studio desktop scope
has the same runtime protection. Existing pages fault in on use; the limit
prevents further swap, without evicting/reloading a model or pausing training.
`deploy/moria/protect_brain_ram.py` verifies the loaded worker's owner, its
dedicated app scope and every member's descent from LM Studio before changing
that scope. Reapply after relaunching LM Studio. Paging is a plausible source
of long-idle delay, not a measured attribution of every prior second.
Codex-Fix for paging: four files, small tier (86 lines), one medium direct
CLI review. Fixed P1 descendant-cgroup validation and P2 optimized-Python
assertion bypass. No verify per the small-tier rule. The deployed helper also
passed with `python3 -O`; both exact scopes report a zero swap limit.

## Reported repeated “Hey Sparks” misses

Matt reported four or five missed “Hey Sparks” wakes. Configured greetings
were only “Hey Spark” and “Hey Sparky”; the plural greeting is now explicitly
included, still requiring “Hey”. The movement stop listener also uses the
configured greetings, and plural greeting removal handles punctuation.
The logs cannot identify all the reported attempts individually.

A separate logged failure woke on “Hey Spark”, decoded only “Joke” at peak
3449, then dropped it as an implausible fragment. The name-only wake chime
blanked 199ms of playback plus 250ms of tail, which can erase command speech.
Voice wakes now acknowledge through eyes and cyan LEDs without playing that
chime or muting the mic, including name-only remote confirmations. Tap-to-talk
keeps its stock chime. “Joke”, “jokes” and “weather” are recognized short
requests inside the already authorized listening window; arbitrary quiet
one-word fragments remain rejected. This does not establish that every wake
miss was caused by the plural greeting or chime.

Validation: 87 focused checks passed. A native Vosk/Parakeet SDK-free probe
accepted a synthetic “Hey Sparks. Tell me a joke” at 100 RMS and rejected
“He sparks a discussion” and “The park is open today”. This synthetic probe
does not measure Matt's actual wake success rate. Codex-Fix scoped five files
(ear, main, config, voice-latency test, this document), small tier: 81 changed
lines, one medium direct CLI review, no findings. Deployed with live brain,
selected voice and saved volume preserved; backup:
`/opt/spark/backups/wake-handoff-20261005-212239/`.

Matt subsequently reported only one in four “Hey Spark” attempts working.
Two name-like segments were rejected before an exact greeting succeeded;
the text logs cannot distinguish missing consonants from ambiguous audio.
A temporary bounded diagnostic in the existing microphone thread records
paired hardware/processed PCM without opening a second microphone or SDK
owner. The first 60-second sample contained no recognizable test greeting
on either path; it cannot establish the cause of the reported wake misses.
The second sample also contained no recognizable greeting. Temporary
instrumentation was removed, diagnostic PCM deleted, and the reviewed
production listener restored healthy at 21:29 EDT. A timed field sample
remains necessary; no further gain or wake-authorization change was made
without one. The one-in-four report remains unresolved.
Codex-Fix for this diagnostic follow-up: documentation only, small tier
(22 lines), one medium direct CLI review, no findings. Production ear/main
hashes match the reviewed local files; health is OK, Gemma brain and Warm
soft robot voice preserved, volume 100.

## October 6: verification coverage and idle paging

Matt reports roughly 20% wake success. At 08:14 the CPU second-opinion
recognizer timed out at two seconds after an overnight idle; its process
had 45MB swapped and Parakeet had 49MB swapped. Paging is a plausible
contributor to that timeout, not proof of the whole reported failure rate.
Both ASR service units now disallow swapping. Deployment restarts and
warms these two ASR services to remove the existing swapped pages.

At 08:28 a successful “Hey Spark, good morning” was followed by Vosk
`[unk]` and Parakeet “Burke, how did you sleep?”. The second recognizer
was never called because the first result didn't match a spelling hint.
Now every eligible failed primary wake check consults the independent
recognizer within the existing total budget. Wake approval still requires
the configured phrase; “Burke” is not an authorized alias.

Opt-in paired audio snapshots use the existing microphone stream, after
high-pass and before/after AEC. A 15-second RAM ring covers wake capture
and ASR delay; snapshots retain the most recent 20 checks, at most 8 seconds
each, in a local private state folder. An absolute live-config deadline
expires capture within ten minutes, including across restarts. Raw audio
is omitted if the processed clip cannot be matched to the ring. No second
ALSA or SDK owner is opened. This is disabled in the repository config.
88 focused checks pass, including actual greeting approval versus room
speech and bounded diagnostic pairing, rotation, unmatched clips and expiry.

Deployed October 6 at 08:36 EDT. ASR backup:
`/opt/whisper-moria/backups/ram-20261006-083601/`; Pi backup:
`/opt/spark/backups/wake-reliability-20261006-083635/`.
Both ASR units are active with current swapped bytes and swap limits zero.
Native SDK-free 100-RMS samples accepted singular/plural greetings and
rejected “Hey Mark” and talk about the park. Primary calls took 0.215–0.219s,
independent calls 0.560–0.580s after restart; these are synthetic tests.
Matt reported all four normal-volume attempts caught after deployment.
The logs also show independent explicit wakes and follow-up requests,
without playback underruns or self-interruption in that test window.
This small field sample does not establish a long-term wake success rate.
Temporary capture was disabled, collected PCM/metadata removed, and the
service returned healthy with diagnostics absent from live config.
Reviewed local/live source hashes matched; Gemma, Warm soft robot and
volume 100 remain intact.

Codex-Fix: large tier, 139 changed lines, seven scoped files: `spark/ear.py`,
`spark/__main__.py`, new `spark/wake_diagnostics.py`,
`deploy/moria/parakeet-server.service`,
`deploy/moria/spark-wake-verifier.service`, `tests/test_voice_latency.py`,
and this document. One medium direct CLI review, no findings; clean exit
without a redundant verify. Prior dirty changes were excluded using fresh
per-file baselines. No ignored findings or fixes from review.

## October 6: dedicated acoustic wake detector

Spark now has a custom CPU ONNX detector for Hey Spark / Hey Sparks /
Hey Sparky, built on openWakeWord speech embeddings. It consumes the
existing processed microphone stream without another SDK or ALSA owner.
The detector can catch a greeting while the existing listener remains a live
backstop, including while the model is healthy but misses a greeting. Eligible
legacy wake checks still call the configured ASR servers. Parakeet still
transcribes requests, and the existing playback echo protection,
interruptions, movement stops, tap controls and stock actions remain.
Load/checksum/inference failures fall back to the legacy wake listener.
Shadow mode logs candidates without changing who decides to wake.

The 2.5-second rolling prefix preserves consumed words. Neural handoff
consumes through that prefix before accepting a silence endpoint, then
extracts the request after an explicitly transcribed greeting, including
when earlier room speech shares the audio. Prefix replay happens once.

Training generated 2,260 unique Piper clips across 109 VCTK speaker IDs
plus four existing voices. Training speakers are separate from evaluation
speakers. Sliding hard-negative and positive windows corrected errors that
end-of-clip tests missed. The evaluation voices were used for development
feedback; this is not a blind benchmark or a real-room success rate.
Final two-frame streaming checks at threshold .95: 162/168 greetings
detected, 0/392 confusing phrases triggered. The separate upstream ambient
features cover 10.697 hours and produced one debounced candidate (.093/hour).
The scripts enforce a 95% synthetic recall and 1% hard-negative ceiling.
Original-to-stream feature comparison matches exactly after warmup.

SDK-free Pi measurement: 65 MB peak process RSS, 17.85 ms p99 and 17.97 ms
maximum per feed, approximately 22% of one CPU core against the 80 ms
inference cadence. Initial load including deterministic silence warmup
took .91 seconds. The robot already has compatible NumPy and ONNX Runtime;
no training dependencies are installed there. Model hashes, attribution,
rebuild and disable instructions live in `models/hey-spark/README.md`.

Installed shadow mode at 09:19 EDT, backup
`/opt/spark/backups/neural-20261006-091959/`. Matt reported both normal-volume
requests answered; logs confirm legacy wake decisions in that test. Shadow
can stop observing when the legacy listener wins early, so that result
does not independently establish the neural detector's field recall.
An info-only cached speaker test completed without self-interruption or
a neural wake. Enabled primary neural mode at 09:23 EDT, backup
`/opt/spark/backups/neural-20261006-092332/`. Health and reviewed source/model
hashes match. Live Gemma, Warm soft robot, volume 100, speech gain 6 dB and
disabled PCM diagnostics are preserved. The repository sample defaults to
shadow mode so a fresh installation does not bypass its field rollout.
Matt reported neither of the two independent primary-listener attempts caught.
Restored shadow mode immediately; on the next pair the legacy recognizer
answered one. The initial shadow success had tested the legacy recognizer,
not independent neural recall, and did not justify removing that backstop.

One short paired microphone sample of the correctly addressed greeting was
captured through the existing microphone and copied to Matt's local training
host. Diagnostic capture was disabled and Pi recording slots deleted after
capture. That recording scores .99 before and after AEC in quiet replay.
Adding preceding speech lowers it below .95, and overlapping speech can lower
it to .001. This reproduces a context sensitivity of the quiet synthetic head;
it does not prove every live miss had the same cause. Additional mixed-room
training remains experimental until its false-trigger and recall gates pass.
The previous model remains the selected bundle; failed candidates are not
installed. The harness now retains the legacy listener in all modes and
silently rejects neural candidates with no transcribed addressed greeting,
so incidental room speech cannot become a command or repeated miss prompts.

Codex-Fix: large tier, 705 changed lines, 12 text files scoped against fresh
baselines (config, requirements, main, ear, neural_wake, voice_latency tests,
this document, three training/evaluation helpers and model manifest/README).
Three binary model hashes were separately verified. Two medium reviews
through direct Codex CLI; success with notes. Fixed P1 shadow-first sample
configuration and P2 manifest/runtime consecutive-frame policy mismatch.
New P2 notes from verification are retained under the skill's two-review
cap: ambient data both calibrates the threshold and supplies its reported
candidate rate, and feed p99 samples include the three non-inference calls
per inference cycle. Consequently the ambient number is a calibration
check, not independent unseen-room performance; feed p99 is a percentile
of all 20ms feeds, not exclusively ONNX calls. The measured worst feed
time (17.97ms) still stays below the 80ms inference interval. No remaining
P0/P1 or additional review loop. No commit or push in this task.

Field-regression follow-up: a final high-effort direct CLI pass found two
P1s in the follow-up changes: greeting-only independent confirmation could
discard the primary decoded command, and the candidate gate still tested
only quiet greetings. Preserved the primary command when confirmation has
no suffix, explicitly accept a leading configured greeting, and added
held-out preceding/overlapping speech cases to the candidate gate. Focused
checks cover neural misses reaching legacy, leading greetings, preserving
a decoded command and silently rejecting room commands. Three review passes
total; no fourth review. Final focused checks: 41 voice, 8 duplex, 4 volume,
2 selected-voice and 10 stock-animation checks pass (65 total).

Final mixed-room candidate: threshold .99 selected on 3.566 calibration
hours; zero events on a disjoint 3.566 evaluation hours, 144/168 quiet
greetings and 268/336 mixed-room greetings detected, zero hard-negative
events in 392 clips. It fails the required 95% recall gates and is not
installed. Its measured inference-call p99 is 2.45ms on Moria. The earlier
mixed versions likewise failed their gates. These results remain development
benchmarks; the selected quiet model's original full ambient result is a
calibration check, not independent field accuracy.

Combined neural plus legacy listening deployed at 09:48 EDT, backup
`/opt/spark/backups/neural-20261006-094823/`. The final reviewed fixes are
applied; source/model hashes and live health verified. Primary-only neural
operation is no longer used. Neural candidates require a transcribed
configured greeting, using the independent recognizer only when needed.
Unconfirmed candidates are silent and do not increment miss prompts.
Another cached speaker check completed without an interruption or underrun.
Temporary human WAVs and metadata were removed from Pi, Moria and local
workspace after diagnosis; diagnostic recording remains off. No human
recording was added to model training. The final combined-listener user
test is pending; no improved real-room success rate is claimed yet.

## Follow-up listening indicator

Green side lights and attentive eyes indicate the actual follow-up listening
window: Matt can reply without the wake phrase. This cue begins after playback
and echo settling, when capture starts. It returns to the normal blue idle
state when capture ends or times out; exceptions also clear the cue. Sleeping
returns to the dark sleepy state. A fresh wake remains cyan, thinking purple
and speaking yellow. No chirp, audio processing or timeout settings change.

Deployed October 6 at 09:58 EDT, with backup
`/opt/spark/backups/followup-status-20261006-095815/`. Live source hashes and
health verified; configuration and volume preserved. All 41 scoped voice
checks passed, and the medium Codex-Fix review found no issues. Matt confirmed
the visual indicator works well.

## Automatic dock warning — October 6

At 13:59 EDT, an automatic idle notice followed sustained negative shunt
readings while parked at 99–100% battery. No recognized user request triggered
it. Near-zero/tapering readings resumed by 14:00. This establishes a temporary
battery discharge interval, not whether the dock cycled or briefly lost contact.
The old announcement overstated that evidence as definitely not charging.

Matt reported a second unsolicited warning at 14:03 while still docked;
the old code was still live. Unsolicited warnings are now suppressed at 95%
battery or above, retaining 15 seconds below that level. Mixed, recovered or unhealthy
readings clear queued warnings; the main thread refreshes evidence before
dispatch. Only a dispatched notice consumes the episode's warning. Wording now
says charging may have paused and asks to check the connection. Motor holds,
charging classification and automatic movement rules are unchanged.

Codex-Fix's first small review noted P2 that an interrupted/failed speech attempt
consumes a warning. This is intentionally one unsolicited speech attempt per
episode; retrying canceled speech would conflict with stopping/repetition control.
Canceled queued warnings do not consume that attempt. The final full-battery
suppression follows Matt's second report, replacing the proposed delay.
Its separate small steering review (52 lines, four files, one medium direct
CLI pass) found P2 that unknown battery telemetry still allowed a warning;
fixed by requiring a known battery below 95%. No verify on either small patch.

Deployed at 14:08 EDT, backup `/opt/spark/backups/charge-notice-20261006-140822/`.
All 31 charging-safety and 41 voice checks passed; local/live source hashes,
health and microphone readiness verified. At 14:08:54, the live monitor reported
100% battery and positive confirmed charging. Configuration/volume preserved.
