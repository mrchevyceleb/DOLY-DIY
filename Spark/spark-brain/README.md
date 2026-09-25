# Spark Brain

A bigger brain for Dolly the Doly robot — LLM-powered conversation on
[Moria's LM Studio](http://192.168.50.204:1234), stock animation playback,
and rollback to stock. This is not full stock application parity.

    mic -> wake word / tap -> VAD -> Parakeet ASR (Moria, :8399)
                              |
                         ROUTER
                  +-----------+-----------+
            stock command match      free speech
                  |                        |
            Doly SDK action       qwen3.6-35b-a3b (Moria)
            (dance/drive/time/..) -> streamed reply -> piper voice (Moria, :8398)

## What's preserved

- Stock doly service stays installed. `spark off` restores it in seconds.
- Stock-style voice commands (time, weather, timer, photo, fist bump,
  drive, dance, colors, sleep, battery, go home) run **locally, instantly, offline**.
- Every Moria dependency degrades gracefully: TTS falls back to on-device
  piper, ASR to on-device whisper/vosk, and commands work with no network.

## Her voice

Default: **Robot** — HFC female +2 semitones + a subtle ring-mod sheen,
synthesized on Moria in ~0.4s (the Pi needs 5-15s for the same model — that
synth latency was most of her 'slow to respond').

The curated picks (Matt auditioned 2026-09, samples in `tools/samples/`):

| say | voice | character |
|---|---|---|
| `switch to robot` | HFC +2st + robot sheen | the default — cute, clear, robotic |
| `switch to hfc` | HFC +2st | same voice, no robot effect |
| `switch to lessac` | Lessac +2st | crispest articulation (in the hopper) |
| `switch to kathleen` | Kathleen +3st | lo-fi toy robot, fastest local synth |

Also available: `cori`, `glados`, `wheatley`, `amy`, `stock`.
Audition new tunings on the robot: `python3 tools/audition_voices.py /tmp/aud`
(edit VARIANTS first; needs `voicefx.py` beside it).

FX are pure stdlib (`spark/voicefx.py`): `tts.pitch_semitones` resamples
(pitch+tempo together — robots talk briskly), `tts.robot_mix` blends a 45 Hz
ring-mod under the dry signal.

## Robot access (Spark / Dolly)

- **Host:** `spark` on the tailnet (`100.112.197.4`), LAN `192.168.50.187`, hostname `doly`
- **SSH:** `ssh doly@spark` — agent key installed (trenzalore's `id_ed25519`)
- **Default credentials** (from the official Doly community post — change if stock image is re-flashed):
  - user `doly` / password `morethanarobot`
  - user `root` / same default password
- **sudo:** passwordless for `doly`
- **Environment:** Raspberry Pi OS bookworm aarch64, Python 3.11.2, ~1.8 GB RAM,
  23 GB free disk, Doly SDK at `/.doly/libs/sdk/lib/python3.11/dist-packages`
  (system python sees it — keep venvs `--system-site-packages`)
- **Stock services:** `doly.service`, `doly_recovery.service`, `doly_update.service`
- **Audio device (mic):** `plughw:CARD=tlv320aic31xxso,DEV=0` (Doly's TLV320AIC31xx codec)
- **Brain server:** LM Studio on Moria `192.168.50.204:1234` — same LAN subnet, direct link
- **Moria services:** piper TTS `:8398`, Parakeet ASR `:8399` — see `deploy/moria/`

## Deploy (on the robot)

    scp -r Spark/spark-brain doly@spark:/home/doly/
    ssh doly@spark
    cd ~/spark-brain && sudo bash deploy/install.sh
    spark test        # text REPL first — verify brain pipeline
    spark on          # go live

Requires: robot on the same LAN as Moria (or tailnet), LM Studio server
running with `qwen/qwen3.6-35b-a3b` loaded, context 8192, TTL off.

## Runtime

- **Say 'Spark'** (or tap her) -> she listens -> speak -> silence ends turn.
- A continuous 150 Hz high-pass filter removes DC/mains hum before either
  recognizer; `audio.highpass_hz: 0` disables it for diagnosis. WebRTC speech
  detection separates speech from fan/room noise for wake
  onset and the end of commands. If Vosk garbles or drops the name, a bounded
  check of the original speech with Moria must confirm "Spark" before
  accepting that turn; unrelated speech cannot execute a command.
  Clear wake words continue to work locally when Moria is unavailable.
  Scheduled idle actions wait for a quiet gap rather than interrupting a
  wake word already being decoded.
- Free speech -> Moria -> streamed reply, spoken sentence by sentence.
- **'Come here' / 'come to me'** -> camera search for one visible person,
  center turns and 40 mm approaches with fresh visual confirmation. Stops
  for target loss, multiple people, edge/proximity/power faults, **'stop'**,
  or a head tap. Bounded to 45 seconds / 600 mm per request; it does not
  identify which person spoke, plan around furniture, or leave its charger.
  Place Spark on a clear floor for this activity. Person detection runs on
  Spark and requires the model installed by `sudo python3 deploy/install_vision.py`.
  The proximity guard accepts fresh VL6180X status 6/7 as no target, as
  documented in [ST DT0020](https://www.st.com/resource/en/design_tip/dt0020-vl6180x-range-status-error-codes-explanation-stmicroelectronics.pdf);
  other faults, stale readings, and measured obstacles within 120 mm stop it.
  Brief sensor interruptions brake first, then require clean readings before
  retrying (at most three retries). Small camera corrections compensate for
  the observed center-turn overshoot; tiny background detections are excluded.
  A wide pose alone does not mean the person is already close. Movement
  follow-up corrections use the controller's result, and chat receives fresh
  power/dock state so older conversation cannot substitute for telemetry.
- **'Dance'** cycles salsa, twist, rock and excited; request **'salsa', 'do the
  twist', 'rock dance', 'party dance', 'work out'** or **'meditate'** by name.
- **'Fist bump' / 'high five'** -> raises arms, lets the ready cue settle, then
  waits up to five seconds for a fresh IMU bump. Proximity and self-motion
  during setup do not count as contact.
- **Petting the top touch pads** -> the three original stock pet reactions,
  with happy eyes, lights and affectionate sounds. Start with a gentle stroke
  lasting about a second; continued strokes build up the reaction. Petting
  reactions accept more strokes instead of cancelling, and coalesce queued
  strokes into the latest level. Touching a moving routine still stops it.
- **'Imagine you are doing exercise' / 'imagine you are at a beach party'**
  -> a brief roleplay line followed by stock choreography. Firefighter,
  police, birthday and meditation themes also select matching routines.
- **'Move forward/back', 'move left/right', 'spin', 'stop'** -> guarded motion.
  Left/right turns 90 degrees; forward/back moves 15 cm. Charging, uncertain
  support and edge detection can block motion. Docked routines retain eyes,
  lights and sound while their motors stay held.
  Voice turns wait for completion before follow-up listening; voice stop and
  head taps can interrupt them. Both front and rear gaps block rotation.
  Requests like "why don't you spin again?" work; complaints like "you didn't
  spin" do not trigger another movement.
- **Voice direction:** camera search is needed because the two
  microphones feed a single ADC in the TLV320AIC3110; Spark's mixer combines
  both inputs into mono. There is no independent left/right signal for voice
  bearing. This was checked against the board schematic and live mixer.
  See [TI's codec specifications](https://www.ti.com/product/TLV320AIC3110).
- **'Go home' / battery at or below 10%** -> requests manual placement while
  `homing.enabled` is false. The camera-guided controller finds the stock
  marker, approaches in short guarded steps, verifies a half-turn with the IMU,
  and backs in at stock speed with arms at 0 degrees. Arrival requires steady
  electrical charging with stopped motors. Centered but oblique or ambiguous
  marker poses are rejected; rear edge events always cancel entry.
  **Home memory (stock-style, `spark/home.py`):** like the stock `HomeControl`,
  the seated charging pose anchors a dead-reckoned world frame (the SDK's
  `get_position` is broken in pybind, so guarded moves credit their own
  estimate). 'Go home' first drives the remembered 450mm standoff blind —
  no marker sighting needed — then the camera controller takes over for
  alignment and entry as above. Each blind leg caps at 1200mm per attempt
  with the usual live edge/power interlocks. A lost pose (pickup, airborne,
  carried off the dock) disables blind navigation and homing falls back to
  the pure visual search; docking again re-anchors the frame.
  See the [manufacturer's homing explanation](https://community.doly.ai/public/d/55-find-home-station-via-sdk).
- **Charging** -> reactions stay parked, including after contact loss. An
  affirmative wheel movement/dance command authorizes a controlled departure.
  With roaming enabled, a minute at 100% with healthy power readings (including
  near-zero charging taper, excluding established discharge) also authorizes
  one departure attempt per dock visit (no repeated nudging on failure).
  Departure starts with one forward 20mm step at speed 25. The stock dock's initial
  two-front-gap profile is allowed only for that step, with electrical proof
  of charging within five seconds; rear gaps remain forbidden in this case.
  It stops within 0.8s if the controller does not complete the step. Any new
  leading gap, touch/voice stop or power fault cancels it. An uncleared front
  gap latches a stop against repeated advances. After supported ground and
  stationary discharge confirm contact release, four guarded 20mm forward
  steps clear the base before turning. During that clearance only, rear-only
  gaps are behind the direction of travel; front and airborne gaps still stop
  immediately. All four
  sensors must see ground and stationary discharge must persist for 1.25s
  before ordinary motion resumes. Arms remain locked during departure.
  A pickup followed by five seconds of supported, discharging placement also
  releases the hold. A direct movement request can verify a stale hold at rest
  and release it without an extra exit step. Lost contact alone never unlocks motors.
  Stops cancel the native drive operation before zeroing both wheels, and
  the power monitor stops unauthorized native operations while docked, and
  independently checks the bounded departure while it is running.
  Power logs include controller states and wheel RPM to diagnose movement.
  The hold survives restarts. Sustained discharge while parked raises one
  spoken reseating notice; battery replies distinguish charging from parking.
  Physical charging is controlled by the hardware, not the Python service.
- **'Go to sleep'** -> quiet standby, no idle routines or follow-up window.
  Say Spark or tap to wake. Sleep never repositions arms or wheels on the dock.
- Casual replies are limited to two sentences / 35 words; explicit requests
  for an explanation allow six sentences / 120 words. The stream closes at
  the limit so long answers do not delay the next listening window.
- **'Search for...' / 'look up...'** -> web search -> brain answers from fresh results.
- **Internet on demand** -> the brain itself decides when it needs the web. Any
  free-speech reply may start with `SEARCH: <query>` (or `READ: <url>` to open a
  page from earlier results); the stream is cut, the tool runs while she says
  "Let me look that up.", and she re-answers with the results injected as
  untrusted context. Her system prompt carries the current date/time so she
  can tell stale knowledge from fresh questions. Zero added latency on turns
  that don't need the web; at most `web.max_hops` (default 2) tool calls per
  reply. `web.enabled: false` in config restores the purely-local brain.
- Moria unreachable -> she says so, **stock commands still work**.

The animation interpreter reads the installed namespace-qualified XML, obeys
the previous-block completion rule, and schedules overlapping sound, lights
and movement on the main thread. Sensor callbacks queue animation requests.
It reports empty files, failed execution and cancellations as failures.
Random ambient beeps remain suppressed; requested routines use their stock audio.
Tap a top touch pad during a performance to cancel it. The main speech loop
resumes after playback; voice interruption during a routine is not implemented.

## Files

| file | role |
|---|---|
| `prompt.md` | Spark personality v4 (source of truth) |
| `config.json` | endpoints, model, VAD thresholds, voice + FX, roam/battery |
| `spark/brain.py` | LM Studio streaming client + fallback model |
| `spark/router.py` | stock-command router + voice switching + weather/timer + web search |
| `spark/search.py` | DuckDuckGo web search (ddgs pkg + HTML fallback), page fetcher, SEARCH/READ tool-call markers |
| `spark/commands.py` | parity table — extend from the Doly app's 'Say' list |
| `spark/body.py` | guarded Doly SDK wrappers (edges, homing, wander, TTS path) |
| `spark/dock_vision.py` | read-only stock dock marker and camera pose diagnostic |
| `spark/ear.py` | arecord + WebRTC speech VAD + wake listener |
| `spark/asr.py` | vosk (wake) + whisper/parakeet HTTP client (utterances) |
| `spark/voicefx.py` | pitch/ring-mod FX for piper WAVs (pure stdlib) |
| `deploy/moria/` | piper TTS + Parakeet ASR servers and systemd units |
| `tools/audition_voices.py` | voice audition rig (samples in `tools/samples/`) |

## Tuning

- False wake / clipped speech -> `audio.start_rms` / `silence_ms` /
  `wake_weak_rms` (loudness floor for acoustic-confusion wake words).
- Swap brains -> `brain.model` (fallback: `brain.fallback_model`).
- Voice -> say `switch to ...` (persists), or set `tts.voice_name` +
  `tts.pitch_semitones` / `tts.robot_mix` in config.
- Roam/battery -> `idle.roam_*`, `idle.low_battery_pct` (default 10),
  `idle.battery_check_s`. Both roaming and homing are currently disabled for
  calibration. The implemented roaming bound is 700mm from the last visual
  dock observation, including a travel margin; it is not a room map. The dock
  must remain visible from the same surface. Pickup invalidates the bound.
- Add stock commands -> append to `COMMANDS` in `spark/commands.py`.
- Web access -> `web.enabled`, `web.max_hops` (tool calls per reply),
  `web.page_max_chars` / `web.page_timeout_s` for `READ:` page fetches.
  `READ:` resolves every host and refuses loopback/LAN/link-local/reserved
  targets, and after a search it only opens URLs that search returned; web
  context is capped at 9000 chars. Accepted residuals (home-robot threat
  model): a DNS-rebinding attacker could race the resolve→connect window, and
  a hung fetch thread is abandoned, not killed (daemon; bounded by usage rate).

## Dock calibration progress

On 2026-09-23, a supervised floor test found that the stock dock's two front
sensors report gaps while seated. A single forward 20mm step at speed 25
cleared both sensors and charging contact. Backward attempts did not exit.
The installed SDK reported zero RPM during that successful forward motion;
neither RPM nor its estimated position is accepted as proof of departure.
Supported ground and sustained stationary discharge are required instead.

The initial 20mm departure controller is live, backed up at
`/opt/spark/backups/departure-20260923-075416`. A stationary check confirmed
`charging=False`, no gaps, and released the persisted hold without movement.
The first voice test exposed a placement hazard latched before charging was
confirmed. A following command-path hardware test successfully left the
contacts but stopped on a front-left gap during its turn near the base.

The new-placement hazard reset and 80mm guarded clearance are now live.
A supervised floor test at 08:11 completed all five 20mm steps and the
requested 90-degree left turn through the normal command router. It ended
with no gaps, sustained discharge, and the dock hold released. The live
files match the tested source; normal microphone listening resumed after
restart. All 58 regression tests pass locally and on the robot. Automatic
full-charge departure is covered in software; this physical test exercised
explicit command departure, not a full-charge idle departure or tabletop use.

The backup is `/opt/spark/backups/dock-clearance-20260923-081059` and the
successful log is `/tmp/spark-dock-clearance-pilot.log`. The earlier failed
turn log is `/tmp/spark-dock-entry-pilot.log`; the diagnostic runner remains
`/tmp/spark-departure-stage/pilot_voice_route.py`. No second SDK process may
run while spark-brain owns the hardware. The temporary floor-probe diagnostic
is not part of the deployed controller.

On 2026-09-22, the camera decoded the physical dock as marker **2** in
`/.doly/data/doly.dict`. Capture at **1280x960**, then resize to the stock
640x480 calibration: native 640x480 capture cropped the dock out of view.
The installed stock detector uses a 30 mm marker side. The diagnostic reads
the installed dictionary and calibration and reports camera-relative position;
its measurements do not authorize motion or bypass edge protection.

With both robot services stopped and Spark safely placed, run on the robot:

    cd /opt/spark/app
    /opt/spark/venv/bin/python -m spark.dock_vision

The diagnostic needs OpenCV with ArUco (validated with 4.13); `--image` accepts
a saved 1280x960 frame without touching hardware.

The floor test in `/tmp/spark-home-pilot10.log` reached the dock and established
charging with the stock arms-at-zero posture and speed 10. The software's
four-second confirmation expired too early, so the stationary window is now
eight seconds. The following `/tmp/spark-home-roundtrip.log` verified departure
and three 60mm roaming steps, then stopped on a rear-left gap after 120mm of
reverse entry. Charging was not established in that round trip. Free roaming
stays disabled until this alignment failure is resolved. Tabletop homing has
not been physically validated.

Subpixel marker corners reduce the angle ambiguity seen in the saved floor
image from roughly +/-9 degrees to +/-2 degrees. Final entry now rejects
any plausible marker pose more than five degrees off perpendicular, and a
transient rear-gap callback latches cancellation even if its GPIO clears.

Profiling also found a 20-second stall in the SDK's Python ToF getter. The
app's `native/tof_reader.cpp` shares the existing sensor instance and releases
the Python GIL during reads. A 150ms acquisition interval avoids starving its
native mutex; motion reads a cache that expires after 300ms. Build with
`deploy/install_tof.py` on the robot. No stock SDK files are replaced.
All 71 regression checks pass on Windows and on the robot staging copy.
These changes were deployed at 08:55 with both feature switches still false.
The live service restarted with every subsystem available; rollback files
are in `/opt/spark/backups/homing-calibration-20260923-085552`.

Further floor calibration found that drive initialization started the shared
IMU with zero offsets, before the later calibrated IMU initialization could
take effect. Drive startup now supplies the factory offsets itself and fails
closed if they cannot be read. Larger docking turns use continuous slow motion
with measured heading, rather than many short starts and stops.

Homing now measures the dock plane from separated camera views. This resolves
some of the small marker's ambiguous poses without repeatedly chasing a flipped
angle. Both IPPE solutions are refined against the observed image corners
before choosing plausible poses; saved-frame checks substantially reduced
reprojection error. Sideways corrections are limited to 30mm, retreat to two guarded 40mm
steps, and translation to 1200mm per attempt. Final alignment requires both
the measured orientation and a fresh visual angle within five degrees.
Entry uses the opposite turn direction to the clockwise dock search.

The centered approach in `/tmp/spark-home-plane3.log` stopped short of the pins;
the user confirmed its position. A single additional 20mm reverse established
charging (`/tmp/spark-entry20-calibrated.log`). The complete standalone return
in `/tmp/spark-home-plane7.log` then established electrical charging without
manual repositioning. Entry therefore includes a bounded 20mm trim, with a
240mm total cap. This does not relax rear-edge or heading stops.

The subsequent leave/roam/return attempt in `/tmp/spark-home-roundtrip3.log`
stopped on visual ambiguity. A later measured-plane approach reached the ramp
but stopped after a six-degree heading change; the user confirmed one track
was catching the dock (`/tmp/spark-home-plane10.log`). Because a single homing
attempt closes at most ~1200mm and an entry misalignment is recoverable,
`go_home` now retries (default 3 attempts, `homing.attempts`) on recoverable
results — limit, lost, not_found, alignment, too_close, turn_unverified —
while deliberate stops, power faults and battery at or below 3 percent do
not retry. Roaming's fixed 800mm ceiling is gone: `roam_radius_mm` (now
3000) is the real bound, and the return trigger rises with distance —
`low_battery_pct` plus `roam_reserve_pct_per_m` (2%/m, capped +10%) from the
anchored roam distance, so she leaves for home before the flat 10 percent
when far away. Full-cycle roaming and homing are ENABLED as of 2026-09-23;
a complete unattended floor round trip is still pending physical validation.
All 83 regression tests pass locally; camera refinement has been checked on
captured robot images.
