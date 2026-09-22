"""Body — Doly SDK wrappers, every subsystem individually guarded.

If a subsystem fails to init (or the module doesn't exist), Spark keeps
running with what works. Availability is tracked in self.has[...].
"""
import os
import re
import sys
import threading
import time
import wave

TTS_WAV = f"/tmp/spark_tts_{os.getuid()}.wav"  # per-UID: service (root) and
# human test sessions (doly) must not fight over one sticky-bit /tmp file
TTS_RAW = f"/tmp/spark_tts_raw_{os.getuid()}.wav"  # synth output before FX


def _log(msg):
    print(f"[body] {msg}", file=sys.stderr)


class Body:
    def __init__(self, cfg, hw=True):
        """hw=False → software-only mode (no SDK init, never stops the doly
        service) — used by --text/--say so testing can't disturb the robot."""
        self.cfg = cfg
        self.hw = hw
        self.has = {}
        self._touch_cb = None
        self._tts_lock = threading.Lock()
        self._cmd_id = 0
        self._muted_sink = None  # text mode prints instead of speaking
        import collections
        self._sfx_queue = collections.deque()
        self._anim_requests = collections.deque()
        # homing: software dead-reckoning (doly_drive.get_position is broken
        # in the pybind layer, so spark tracks its own estimate)
        self._pose = None          # [x_mm, y_mm, heading_deg] or None = unknown
        self._homing = False
        self._leaving_home = False
        self._home_arrived = False
        self._edge_hazard = None   # None | "forward" | "backward" | "all":
        # latched after a real gap event — no blind drives TOWARD that edge
        # until she escapes it or it is verifiably gone
        self._hazard_clear_at = None
        self._hazard_airborne = False  # an all-void event followed the latch
        self._escaping = False      # _escape_edge drives bypass the hazard gate
        self.docked = False
        self._dock_clear_since = None
        self._power_stop = threading.Event()
        self._power_thread = None
        self._power_fault = False
        self._power_lock = threading.RLock()
        self._charging = None
        if hw:
            from .charging import ChargingMonitor
            self._charging = ChargingMonitor()
        self._hazard_lock = threading.Lock()
        self._hazard_gen = 0        # bumped on every latch; clears check it
        self.anim = None           # AnimPlayer, created in _init_all
        if self.hw:
            self._init_all()

    def _next_id(self):
        self._cmd_id = (self._cmd_id % 65535) + 1
        return self._cmd_id

    # ------------------------------------------------------------- init chain
    def _init_all(self):
        try:
            import doly_helper as helper
            rc = helper.stop_doly_service()
            if rc < 0:
                # doly service still running = hardware conflicts ahead;
                # do NOT claim readiness (examples abort here)
                raise RuntimeError(f"stop_doly_service failed rc={rc}")
            self._helper = helper
            self.has["helper"] = True
        except Exception as e:
            _log(f"helper unavailable: {e} — hardware will likely conflict")

        if self.has.get("helper"):
            try:
                if self._helper.read_settings() < 0:
                    _log("read_settings failed (arm calibration may be off)")
            except Exception as e:
                _log(f"read_settings: {e}")

        self._try("sound", self._init_sound)
        self._try("tts", self._init_tts)
        self._try("eye", self._init_eye)
        self._try("touch", self._init_touch)
        self._try("arm", self._init_arm)
        self._try("drive", self._init_drive)
        self._try("led", self._init_led)
        self._try("battery", self._init_battery)
        self._try("edge", self._init_edge)
        self._try("tof", self._init_tof)
        self._try("imu", self._init_imu)
        self.react_enabled = True
        self._react_debounce = {}
        from .anim import AnimPlayer
        self.anim = AnimPlayer(self, self.cfg)
        self._start_power_monitor()
        _log(f"online subsystems: {[k for k, v in self.has.items() if v]}")

    def _try(self, name, fn):
        try:
            fn()
            self.has[name] = True
        except Exception as e:
            self.has[name] = False
            _log(f"{name} unavailable: {e}")

    def _init_sound(self):
        import doly_sound as snd
        rc = snd.init()
        if rc < 0:
            raise RuntimeError(f"sound init rc={rc}")
        try:
            snd.set_volume(int(self.cfg.get("sounds", {}).get("volume", 90)))
        except Exception:
            pass
        self._snd = snd

    def _init_tts(self):
        import doly_tts as tts
        voice = getattr(tts.VoiceModel, f"Model{int(self.cfg['tts']['voice_model'])}",
                        tts.VoiceModel.Model1)
        rc = tts.init(voice, TTS_WAV)
        if rc < 0:
            raise RuntimeError(f"tts init rc={rc}")
        self._tts = tts

    def _init_eye(self):
        import doly_eye as eye
        from doly_color import ColorCode
        rc = eye.init(ColorCode.Blue, ColorCode.White)
        if rc != 0:
            raise RuntimeError(f"eye init rc={rc}")
        eye.on_start(lambda i: None)
        eye.on_complete(lambda i: None)
        eye.on_abort(lambda i: None)
        self._eye, self._colorcode = eye, ColorCode

    def _init_touch(self):
        import doly_touch as touch
        rc = touch.init()
        if rc < 0:
            raise RuntimeError(f"touch init rc={rc}")

        def _on_touch(side, state):
            if self._touch_cb:
                try:
                    self._touch_cb(side, state)
                except Exception as e:
                    _log(f"touch cb error: {e}")

        touch.on_touch(_on_touch)
        self._touch = touch

    def _init_arm(self):
        import doly_arm as arm
        rc = arm.init()
        if rc < 0:
            raise RuntimeError(f"arm init rc={rc}")
        self._arm = arm

    def _init_drive(self):
        import doly_drive as drive
        rc = drive.init()
        if rc != 0:
            raise RuntimeError(f"drive init rc={rc}")
        drive.on_complete(lambda i: None)
        drive.on_error(lambda i, s, t: _log(f"drive error {i}"))
        self._drive = drive

    def _init_led(self):
        import doly_led as led
        from doly_color import Color, ColorCode
        rc = led.init()
        if rc != 0:
            raise RuntimeError(f"led init rc={rc}")
        led.on_complete(lambda i: None)
        led.on_error(lambda i, s, t: None)
        self._led, self._color, self._ledcc = led, Color, ColorCode

    def _init_battery(self):
        import doly_battery as bat
        rc = bat.init()
        if rc < 0:
            raise RuntimeError(f"battery init rc={rc}")
        self._battery = bat

    # -------------------------------------------------------- sensor reactions
    def _init_tof(self):
        """Proximity / hand gestures via ToF (stock 'proximity sense')."""
        import doly_tof as tof
        rc = tof.init()
        if rc < 0:
            raise RuntimeError(f"tof init rc={rc}")

        def _on_gesture(left, right):
            for s in (left, right):
                t = str(getattr(s, "type", "")).split(".")[-1]
                if t and t != "Undefined":
                    self._sensor_react("tof", t)

        tof.on_proximity_gesture(_on_gesture)
        if tof.setup_continuous(50, 60) < 0:
            raise RuntimeError("tof setup_continuous failed")
        self._tof = tof

    def _init_imu(self):
        """Bump / poke / shake / lift awareness (stock physical reactions)."""
        import doly_imu as imu
        rc, gx, gy, gz, ax, ay, az = self._helper.get_imu_offsets()
        if rc < 0:
            raise RuntimeError("imu offsets unavailable")
        rc = imu.init(1, gx, gy, gz, ax, ay, az)
        if rc < 0:
            raise RuntimeError(f"imu init rc={rc}")

        def _on_gesture(gesture, direction):
            g = str(gesture).split(".")[-1]
            self._sensor_react("imu", g)

        def _on_update(data):
            try:
                self._imu_yaw = data.ypr.yaw
            except Exception:
                pass

        imu.on_gesture(_on_gesture)
        try:
            imu.on_update(_on_update)
        except Exception:
            pass
        self._imu = imu

    _TOF_REACTIONS = {
        "ObjectComing": ("CAUTIOUS", None, None),      # eyes only — walk-by
        # clicking drove Matt up the wall; NEVER auto-drive either
        "ObjectGoing": ("HAPPY", None, None),            # (see table-fall postmortem)
        "Scrubing": ("SPARKLING", "pet", None),
        "ToLeft": ("LOOK_LEFT", None, None),
        "ToRight": ("LOOK_RIGHT", None, None),
    }
    _IMU_REACTIONS = {
        "ShockLight": ("BUMP", None, None),            # desk bumps: eyes only,
        "ShockMedium": ("BUGGED", None, None),         # no random clicks/damage
        "ShockHard": ("DAMAGED", "alarm", None),
        "ShockExtreme": ("DESTROYED", "alarm", None),
        "ShortShake": ("DIZZY_L", "debuff", None),
        "LongShake": ("DIZZY_R", "debuff", None),
        "Vibrate": ("NERVOUS", None, None),
        # sustained desk vibration is NOT a crash — the alarm was the
        # random BEEP-BEEP-BEEO Matt kept hearing. Eyes only.
        "VibrateExtreme": ("FRIGHTENED", None, None),
        "Move": ("LOOK_AHEAD", None, None),
    }

    def _sensor_react(self, family, kind):
        """Thread-safe, debounced stock-style reactions. Never raises."""
        try:
            if not getattr(self, "react_enabled", True):
                return
            now = time.time()
            key = f"{family}:{kind}"
            # "Move" fires on plate vibration every few seconds — throttle it
            # hard or she stares LOOK_AHEAD all day instead of idling calmly
            debounce = 30.0 if kind == "Move" else 3.0
            if now - self._react_debounce.get(key, 0) < debounce:
                return
            self._react_debounce[key] = now
            table = self._TOF_REACTIONS if family == "tof" else self._IMU_REACTIONS
            expr, sfx, motion = table.get(kind, (None, None, None))
            if sfx and (getattr(self, "docked", False) or len(self._edge_gaps()) >= 4):
                sfx = None  # docked = quiet time: eyes react, mouth stays shut
            _log(f"react {key}: expr={expr} sfx={sfx} motion={motion}")
            if sfx:
                sfx_path = self.cfg.get("sounds", {}).get("sfx_map", {}).get(sfx)
                if sfx_path:
                    self.play_sfx(sfx_path)
            if expr and self.has.get("eye"):
                try:
                    self._eye.set_animation(
                        self._next_id(), getattr(self._eye.expressions, expr))
                except Exception:
                    pass
            if motion == "back":
                self.drive_distance(-100, speed=35)
            elif motion == "forward":
                self.drive_distance(80, speed=35)
        except Exception as e:
            _log(f"react failed: {e}")

    def mood_eyes(self, expression_name):
        """Set a specific expression by enum name (used by petting logic)."""
        if not self.has.get("eye"):
            return
        try:
            self._eye.set_animation(
                self._next_id(), getattr(self._eye.expressions, expression_name))
        except Exception as e:
            _log(f"mood_eyes failed: {e}")

    def _init_edge(self):
        """ToF edge sensors — the not-driving-off-tables subsystem."""
        import doly_edge as edge
        rc = edge.init()
        if rc < 0:
            raise RuntimeError(f"edge init rc={rc}")

        def _handle_gap(direction):
            dir_name = str(direction).split(".")[-1]
            # EMERGENCY: kill motion, lock further motion, react
            lock_s = 10.0 if dir_name == "All" else 3.0  # All = airborne/off-edge
            self._gap_lock_until = max(self._gap_lock_until, time.time() + lock_s)
            self._hazard_clear_at = None  # any gap event resets the quiet clock
            if not getattr(self, "docked", False):
                # real cliff/airborne event (not plate noise): latch the
                # hazard's direction so the next blind drive can't re-run
                # the same edge. "All" off the dock = carried or falling:
                # block everything, and remember it as pick-up evidence
                # (a later all-void is what proves a relocation happened).
                if dir_name == "All":
                    had_latch = getattr(self, "_edge_hazard", None) is not None
                    self._latch_hazard("all")
                    if had_latch:
                        # an all-void on TOP of an existing latch = picked
                        # up / relocated (the fast-clear evidence). The
                        # fall that CREATES the latch is not evidence.
                        self._hazard_airborne = True
                elif len(self._edge_gaps()) < 3:
                    if dir_name.startswith("Front"):
                        self._latch_hazard("forward")
                    elif dir_name.startswith("Back"):
                        self._latch_hazard("backward")
            _log(f"GAP DETECTED dir={dir_name} — motion locked {lock_s}s")
            if dir_name == "All":
                self._pose = None          # airborne/picked up: position lost
            try:
                self.drive_stop()
            except Exception:
                pass
            self.eyes("thinking")
            self._led_flash("Red")
            if dir_name == "All":
                self.mood_eyes("FRIGHTENED")

        def _on_gap(direction):
            with self._power_lock:
                _handle_gap(direction)

        edge.on_gap_detect(_on_gap)
        rc = edge.enable_control()
        if rc < 0:
            raise RuntimeError(f"edge enable_control rc={rc}")
        self._gap_lock_until = 0.0
        self._edge = edge

    # ------------------------------------------------------------------- TTS
    def _produce_speech(self, text):
        """Synthesize text into TTS_WAV.

        Priority: Moria's piper server (tts.server_url — ~0.4s vs 5-15s on
        the Pi) -> local piper (tts.piper_model) -> stock doly_tts.
        Voice FX (pitch_semitones / robot_mix) always apply on top, so
        Spark sounds identical regardless of where the synth ran."""
        tts_cfg = self.cfg.get("tts", {})
        produced = False
        server = tts_cfg.get("server_url")
        if server and tts_cfg.get("voice_name") is not None:
            try:
                self._produce_server(text, server, tts_cfg)
                produced = True
            except Exception as e:
                _log(f"server TTS failed ({e}) — falling back to local synth")
        if not produced:
            model = tts_cfg.get("piper_model")
            try:
                if model:
                    self._produce_piper(text, model)
                    produced = True
            except Exception as e:
                _log(f"local piper failed ({e}) — falling back to stock voice")
        if not produced:
            self._tts.produce(text)
            return  # stock voice writes TTS_WAV itself; no FX on it
        try:
            self._apply_fx()
        except Exception as e:
            _log(f"voice FX failed ({e}) — using raw synth")
            import shutil
            shutil.copyfile(TTS_RAW, TTS_WAV)

    def _produce_server(self, text, server, tts_cfg):
        """Moria's piper HTTP server: POST plain text, receive WAV."""
        import urllib.request
        import urllib.parse
        voice = tts_cfg.get("voice_name")
        url = server.rstrip("/") + "/"
        if voice:
            url += "?" + urllib.parse.urlencode({"voice": voice})
        req = urllib.request.Request(url, data=text.encode("utf-8"))
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status != 200:
                raise RuntimeError(f"server rc={resp.status}")
            data = resp.read(4 * 1024 * 1024 + 1)  # hard cap: sentences, not novels
        if len(data) < 100 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise RuntimeError(f"response is not a WAV ({len(data)} bytes)")
        import io
        import wave as _wave
        with _wave.open(io.BytesIO(data), "rb") as w:  # structural parse, not just magic
            if w.getnframes() == 0:
                raise RuntimeError("server returned an empty WAV")
        with open(TTS_RAW, "wb") as f:
            f.write(data)
        _log(f"server TTS {time.time()-t0:.2f}s ({len(data)} bytes)")

    def _apply_fx(self):
        """Run voicefx over the raw synth when FX are configured."""
        tts_cfg = self.cfg.get("tts", {})
        pitch = float(tts_cfg.get("pitch_semitones", 0) or 0)
        robot = float(tts_cfg.get("robot_mix", 0) or 0)
        if pitch or robot:
            from . import voicefx
            voicefx.apply_fx(TTS_RAW, TTS_WAV, pitch_semitones=pitch, robot_mix=robot)
        else:
            import shutil
            shutil.copyfile(TTS_RAW, TTS_WAV)

    def _produce_piper(self, text, model):
        """Synthesize via piper: the stock binary by default, or the modern
        piper-tts module (tts.piper_module_python) for voices whose phoneme
        maps need multi-codepoint support (e.g. Wheatley)."""
        import subprocess
        tts_cfg = self.cfg.get("tts", {})
        py = tts_cfg.get("piper_module_python")
        if py:
            cmd = [py, "-m", "piper", "--model", model,
                   "--output-file", TTS_RAW]
            if tts_cfg.get("piper_length_scale"):
                cmd += ["--length-scale", str(tts_cfg["piper_length_scale"])]
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True,
                                  timeout=30)
        else:
            cmd = [tts_cfg.get("piper_bin", "/.doly/libs/piper/lib/piper"),
                   "--model", model,
                   "--espeak_data", tts_cfg.get("piper_espeak_data",
                                                "/.doly/libs/piper/lib/espeak-ng-data"),
                   "--output_file", TTS_RAW, "-q"]
            if tts_cfg.get("piper_length_scale"):
                cmd += ["--length_scale", str(tts_cfg["piper_length_scale"])]
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = "/.doly/libs/piper/lib:" + env.get("LD_LIBRARY_PATH", "")
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True,
                                  timeout=30, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"piper rc={proc.returncode}: {proc.stderr[:120]}")

    def speak(self, text, wait=True):
        """Say something out loud (piper voice if configured, else stock).
        Returns False if muted."""
        text = (text or "").strip()
        if not text:
            return True
        if not (self.has.get("tts") and self.has.get("sound")):
            if self._muted_sink:
                self._muted_sink(text)
            else:
                _log(f"(muted) {text}")
            return False
        with self._tts_lock:
            try:
                started = time.perf_counter()
                # strip anything the synth would read literally
                text = re.sub(r"[*_`#>]+", "", text)
                self._produce_speech(text)
                self._snd.play(TTS_WAV, self._next_id())  # (file, block_id)
                dur = self._wav_duration(TTS_WAV)
                self._speaking_until = time.time() + dur + 0.25
                _log(f"speech: first audio {time.perf_counter()-started:.2f}s")
                if wait:
                    time.sleep(dur + 0.15)
                return True
            except Exception as e:
                _log(f"speak failed: {e}")
                return False

    @staticmethod
    def _wav_duration(path):
        try:
            with wave.open(path, "rb") as w:
                return w.getnframes() / float(w.getframerate() or 16000)
        except Exception:
            return 2.0

    def wake_reaction(self, audible=True):
        """'Hey Spark' acknowledged: stock wake chirp + WAKE_WORD eyes + cyan.

        defer=False: this runs on the main thread (right after the wake
        listener returns), so direct playback is GIL-safe — the deferred
        queue wouldn't flush until next turn and the chirp would be silent.
        """
        chirp = self.cfg.get("wake", {}).get("chirp")
        if chirp and audible:
            self.play_sfx(chirp, defer=False)
        self.mood_eyes("WAKE_WORD")
        self._led_flash("Cyan")

    def speaking_recently(self):
        """True while our own TTS output might still reach the mic."""
        return time.time() < getattr(self, "_speaking_until", 0)

    def speak_stream(self, sentences):
        """Pipelined TTS: synthesize sentence N+1 while sentence N plays.

        First word still waits for the first synth, but multi-sentence
        replies no longer serialize synth+play per sentence.
        """
        if not (self.has.get("tts") and self.has.get("sound")):
            for sent in sentences:
                self.speak(sent, wait=False)
            return

        import shutil
        prev_end = 0.0
        played = []
        started = time.perf_counter()
        with self._tts_lock:
            try:
                # Consume lazily: requesting the whole list buffers the LLM.
                for i, sent in enumerate(sentences):
                    text = re.sub(r"[*_`#>]+", "", (sent or "").strip())
                    if not text:
                        continue
                    tmp = f"/tmp/spark_tts_{os.getuid()}_{i}.wav"
                    self._produce_speech(text)
                    shutil.copyfile(TTS_WAV, tmp)
                    played.append(tmp)
                    time.sleep(max(0, prev_end - time.time()))
                    self._snd.play(tmp, self._next_id())
                    prev_end = time.time() + self._wav_duration(tmp) + 0.05
                    self._speaking_until = prev_end + 0.20
                    if len(played) == 1:
                        _log(f"stream: first audio {time.perf_counter()-started:.2f}s")
            finally:
                # A broken LLM stream must still finish/clean up queued audio.
                time.sleep(max(0, prev_end - time.time()))
                for tmp in played:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def pet_pulse(self):
        """Instant 'I felt that' reaction: sfx chirp + LED flash + happy eyes."""
        sfx = self.cfg.get("sounds", {}).get("pet_sfx")
        if sfx:
            self.play_sfx(sfx)
        self._led_flash("Cyan")
        self.eyes("listening")

    # ---------------------------------------------------------- dock sensing
    def _start_power_monitor(self):
        def watch():
            while not self._power_stop.is_set():
                self.refresh_power()
                self._power_stop.wait(0.25)
        self._power_thread = threading.Thread(target=watch, daemon=True)
        self._power_thread.start()

    def refresh_power(self):
        """Serialize fault state with telemetry, motor checks and dispatch."""
        with self._power_lock:
            try:
                result = self._refresh_power()
                fault = result is None
                if fault and not self._power_fault:
                    self._power_fault = True
                    self.stop_everything()  # stop arms already in flight too
                self._power_fault = fault
                return result
            except Exception as e:
                self._power_fault = True
                _log(f"power monitor failed: {e}; motors held")
                try:
                    self.stop_everything()
                except Exception:
                    pass
                return None

    def _refresh_power(self):
        """Charging latches a motor hold. Lost contact never triggers a nudge.
        Release only after sustained discharge on fully supported ground."""
        if self._charging is None:
            return None
        with self._power_lock:
            charging = self._charging.sample()
            now = time.monotonic()
            if charging and not self._leaving_home:
                if not self.docked:
                    self.docked = True
                    self._pose = None
                    self.stop_everything()
                    _log("charging confirmed: motors parked")
                self._dock_clear_since = None
            elif self.docked and not self._leaving_home:
                clear = (charging is False and self._charging.average is not None
                         and self._charging.average < -5
                         and self.has.get("edge") and not self._edge_gaps())
                if clear:
                    if self._dock_clear_since is None:
                        self._dock_clear_since = now
                    elif now - self._dock_clear_since >= 5:
                        self.docked = False
                        self._pose = None
                        self._dock_clear_since = None
                        _log("removed from charger: supported ground confirmed")
                else:
                    self._dock_clear_since = None
            if now >= getattr(self, "_next_power_log", 0):
                self._next_power_log = now + 30
                _log(f"power: battery={self.battery_pct()}% charging={charging} "
                     f"parked={self.docked} shunt={self._charging.average} "
                     f"voltage={self._charging.voltage} error={self._charging.error}")
            return charging

    def dock_probe(self):
        """Read charging telemetry; never move wheels to test the dock."""
        self.refresh_power()
        return self.docked

    def actuators_held(self):
        """Keep arms and wheels still on charge or with uncertain power."""
        if not self.hw:
            return False
        charging = self.refresh_power()
        pct = self.battery_pct()
        if pct is None or pct <= 2 or charging is None or self._power_fault:
            return True
        # A full battery can taper to zero current. Ambiguous support must
        # also HOLD motion, never serve as permission to cross a plate lip.
        if not self.has.get("edge") or len(self._edge_gaps()) >= 2:
            return True
        return self.docked

    def ensure_mobility(self):
        return not self.actuators_held()

    # ------------------------------------------------------------- edge lock
    def _edge_gaps(self):
        """Which sensors currently see a void (['Front_Left', 'Back_Right', ...])."""
        if not self.has.get("edge"):
            return []
        try:
            state = getattr(self._edge.GpioState,
                            self.cfg.get("edge", {}).get("gap_gpio_state", "Low"))
            gaps = [str(getattr(s, "id", "?")).split(".")[-1]
                    for s in self._edge.get_sensors(state)]
        except Exception as e:
            _log(f"preflight poll failed: {e}")
            gaps = ["Front_Left", "Front_Right", "Back_Left", "Back_Right"]
        return gaps

    def _motion_allowed(self, direction="forward"):
        """Direction-aware safety: a cliff BEHIND her must not block forward
        motion (that bug trapped her on the dock and froze her near desk
        edges). forward -> only Front gaps block; backward -> only Back gaps;
        rotate -> Front gaps block (Back-only tolerated; watchdog guards
        mid-rotation sweeps).
        """
        if self.actuators_held():
            return False
        if not self.has.get("edge"):
            return False
        if time.time() < getattr(self, "_gap_lock_until", 0):
            _log(f"motion blocked ({direction}): gap lock active")
            return False
        gaps = self._edge_gaps()
        if not gaps:
            return True
        front = any(g.startswith("Front") for g in gaps)
        back = any(g.startswith("Back") for g in gaps)
        blocked = (front and direction in ("forward", "rotate")) or \
                  (back and direction == "backward")
        if blocked:
            self._gap_lock_until = time.time() + 2.0
            _log(f"motion blocked ({direction}): gaps={gaps}")
            return False
        _log(f"preflight pass ({direction}): tolerating {gaps}")
        return True

    def _latch_hazard(self, new):
        """Merge, never overwrite: a rear cliff must not erase the front
        one. Any disagreement (or an airborne event) becomes "all" — boxed
        in, both directions blocked until verifiably clear. Thread-safe:
        sensor events race command-path clears, so every latch bumps a
        generation counter that clears must re-verify."""
        with self._hazard_lock:
            cur = self._edge_hazard
            if cur in (None, new):
                self._edge_hazard = new
            else:
                self._edge_hazard = "all"
            self._hazard_clear_at = None  # fresh evidence restarts the clock
            self._hazard_gen += 1

    def _hazard_active(self, direction="forward"):
        """Edge hazard latch: True while motion TOWARD a cliff she already
        hit is unsafe. The latch is directional — a rear cliff must not
        block the forward escape route. Charging and gap counts never
        exempt a latch. Clears when the edge is verifiably gone or when
        _escape_edge completes."""
        latch = getattr(self, "_edge_hazard", None)
        if not latch or latch not in ("all", direction):
            return False
        gaps = self._edge_gaps()
        if not gaps:
            # zero gaps alone do not prove she was moved somewhere safe —
            # she could be parked just short of the same lip. Fast-clear
            # only with pick-up evidence (an all-void event AFTER the
            # latch); otherwise hold for a long quiet spell and let a
            # verified _escape_edge be the normal way out. Arms and clears
            # are generation-checked so a fresh async latch always wins.
            hold_s = 5.0 if getattr(self, "_hazard_airborne", False) else 60.0
            armed = getattr(self, "_hazard_clear_at", None)
            if armed is None:
                self._hazard_clear_at = (getattr(self, "_hazard_gen", 0),
                                         time.time() + hold_s)
                return True
            gen_at_arm, deadline = armed
            if gen_at_arm != getattr(self, "_hazard_gen", 0):
                self._hazard_clear_at = None  # stale arm from older evidence
                return True
            if time.time() >= deadline:
                with self._hazard_lock:
                    if self._hazard_gen == gen_at_arm:
                        self._edge_hazard = None
                        self._hazard_clear_at = None
                        self._hazard_airborne = False
                        _log("edge hazard cleared")
                        return False
                return True  # a fresh hazard raced the clear — it wins
            return True
        self._hazard_clear_at = None
        return True

    def drive_guarded(self, mm, speed=25, segment_mm=60):
        """Segmented drive with INLINE edge polling — the 'come here' fix.

        Blind 250mm drives at speed 45 put her off the desk: the async
        watchdog saw the gap but momentum won. Now motion is chopped into
        <=60mm segments at a sane speed, each preflighted, with edge gaps
        polled every 30ms DURING the segment. First danger sign: hard stop
        and latch the hazard. Returns 'ok' | 'stopped_edge' | False."""
        import math
        try:
            mm, speed, segment_mm = float(mm), float(speed), float(segment_mm)
        except (TypeError, ValueError):
            return False
        if not all(map(math.isfinite, (mm, speed, segment_mm))) or \
                mm == 0 or speed <= 0 or segment_mm <= 0:
            _log(f"guarded drive rejected: bad args mm={mm} speed={speed} seg={segment_mm}")
            return False
        if speed > 30.0 or segment_mm > 60.0:
            # the whole point is short, slow segments — no caller may
            # recreate the blind 250mm@45 fall drive through this API
            _log(f"guarded drive: clamping speed={speed} seg={segment_mm}")
            speed = min(speed, 30.0)
            segment_mm = min(segment_mm, 60.0)
        if abs(mm) > 500.0 or speed < 10.0 or segment_mm < 20.0:
            # operational bounds: huge distances = unbounded blocking loop,
            # near-zero speed = motor stall, tiny segments = command spam
            _log(f"guarded drive rejected: out of bounds mm={mm} speed={speed} seg={segment_mm}")
            return False
        if not self.has.get("drive"):
            return False
        direction = "forward" if mm >= 0 else "backward"
        if self._hazard_active(direction) and not (self._leaving_home or self._escaping):
            _log(f"guarded drive refused: edge hazard latched ({direction})")
            return False
        if not self.ensure_mobility() or not self._motion_allowed(direction):
            _log("guarded drive blocked: parked, low power, or edge")
            return False
        remaining = abs(mm)
        total = remaining
        sign = 1 if mm >= 0 else -1
        while remaining > 0:
            step = min(segment_mm, remaining)
            if not self._motion_allowed(direction):
                return "stopped_edge" if remaining < total else False
            try:
                # pybind11 SupportsInt rejects floats — step/speed are
                # floats after the clamp math above and must be reified
                with self._power_lock:
                    if (not self._motion_allowed(direction)
                            or (self._hazard_active(direction) and not self._escaping)):
                        return False
                    self._drive.go_distance(self._next_id(), int(round(step)),
                                            int(round(speed)), sign > 0, True)
            except Exception as e:
                self.drive_stop()  # a half-sent command must not free-run
                _log(f"guarded drive failed: {e}")
                return False
            saw_running = False
            completed = False
            start = time.time()
            time.sleep(0.05)  # let the drive enter Running before polling
            deadline = start + 8
            while time.time() < deadline:
                if self.actuators_held():
                    self.drive_stop()
                    return False
                gaps = self._edge_gaps()
                if gaps and not self.is_on_dock():
                    front = any(g.startswith("Front") for g in gaps)
                    back = any(g.startswith("Back") for g in gaps)
                    if (front and direction == "forward") or (back and direction == "backward"):
                        self.drive_stop()
                        self._latch_hazard(direction)
                        _log(f"guarded drive: stopped at edge ({direction}, gaps={gaps})")
                        return "stopped_edge"
                try:
                    st = self._drive.get_state()
                except Exception:
                    break  # comms error: do NOT credit the segment
                DS = self._drive.DriveState
                if st == DS.Running:
                    saw_running = True
                elif st == DS.Completed and saw_running:
                    completed = True  # explicit terminal success
                    break
                elif st == DS.Error:
                    _log("guarded drive: controller reported Error")
                    break
                elif not saw_running and time.time() - start > 1.0:
                    break  # never started: rejected/stale command
                time.sleep(0.03)
            self.drive_stop()
            if not completed:
                _log("guarded drive: segment incomplete (stall/timeout/comms/rejected) — aborting")
                return False
            # a watchdog stop can look exactly like completion — never
            # credit a segment that ended facing a hazard
            gaps = self._edge_gaps()
            if gaps and not self.is_on_dock():
                front = any(g.startswith("Front") for g in gaps)
                back = any(g.startswith("Back") for g in gaps)
                if (front and direction == "forward") or (back and direction == "backward"):
                    self._latch_hazard(direction)
                    _log(f"guarded drive: stopped at edge after segment ({direction}, gaps={gaps})")
                    return "stopped_edge"
            self._pose_update(dist_mm=sign * step)  # credit only real travel
            remaining -= step
        return "ok"

    def _watch_motion(self, direction="forward"):
        """Stop active drives on loss of power permission or an edge."""
        def _run():
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    if self._leaving_home:
                        return  # bounded undock polls inline
                    if self.actuators_held():
                        self.drive_stop()
                        return
                    if self._drive.get_state() != self._drive.DriveState.Running:
                        return
                    gaps = self._edge_gaps()
                    if gaps:
                        front = any(g.startswith("Front") for g in gaps)
                        back = any(g.startswith("Back") for g in gaps)
                        danger = (front and direction in ("forward", "rotate")) or \
                                 (back and direction == "backward")
                        if danger:
                            self._gap_lock_until = max(self._gap_lock_until, time.time() + 3.0)
                            if not self.is_on_dock():
                                # a rotate-stop found the cliff with a swept
                                # wheel: latch forward so _escape_edge keeps
                                # the backward retreat route open
                                self._latch_hazard(direction if direction in ("forward", "backward") else "forward")
                            self.drive_stop()
                            _log(f"watchdog: stopped mid-motion ({direction}, gaps={gaps})")
                            return
                    time.sleep(0.03)
            except Exception as e:
                _log(f"watchdog error: {e}")
        if self.has.get("edge") and self.has.get("drive"):
            threading.Thread(target=_run, daemon=True).start()

    def _led_flash(self, color_name):
        if not self.has.get("led"):
            return
        try:
            activity = self._led.LedActivity()
            activity.mainColor = self._color.from_code(getattr(self._ledcc, color_name, self._ledcc.Red))
            activity.fadeColor = self._color.from_code(self._ledcc.Black)
            activity.fade_time = 300
            self._led.process_activity(self._next_id(), self._led.LedSide.Both, activity)
        except Exception as e:
            _log(f"led_flash failed: {e}")

    def play_sfx(self, path, defer=True):
        """Play a sound effect. Foreign (native callback) threads pass
        defer=True — the file is queued and flushed by the main loop, so
        snd.play() never runs concurrently with Vosk decoding (GIL abort)."""
        if not self.has.get("sound"):
            return False
        if not path:
            return False
        if defer:
            self._sfx_queue.append(path)
            return True
        try:
            self._snd.play(path, self._next_id())
            return True
        except Exception as e:
            _log(f"play_sfx failed: {e}")
            return False

    def flush_sfx(self):
        """Main-loop drain: actually play queued sfx (safe GIL context)."""
        while self._sfx_queue:
            path = self._sfx_queue.popleft()
            try:
                self._snd.play(path, self._next_id())
            except Exception as e:
                _log(f"flush_sfx failed: {e}")
    # real doly_eye.expressions members (verified on-robot 2024 image)
    # ------------------------------------------------------------------ eyes
    _EXPR_CANDIDATES = {
        "listening": ["ATTENTION", "WAKE_WORD", "LOOK_AHEAD"],
        "thinking": ["SCAN", "CONCENTRATE", "THINK"],
        "speaking": ["HAPPY", "CHEERFUL", "EXCITED"],
        "idle": ["BLINK", "FINE", "BLINK_ONLY"],
        "sleepy": ["SLEEPY", "SLEEP", "DROWSY"],
    }

    _LED_MOODS = {  # (color, fade ms) — ambient moods per state
        "listening": ("Cyan", 600),
        "thinking": ("Purple", 900),
        "speaking": ("Yellow", 500),
        "idle": ("Blue", 2000),
        "sleepy": ("Black", 1500),
    }

    def eyes(self, state):
        if not self.has.get("eye"):
            return
        try:
            exprs = getattr(self._eye, "expressions", None)
            if exprs is None:
                return
            chosen = None
            for name in self._EXPR_CANDIDATES.get(state, ["NORMAL"]):
                chosen = getattr(exprs, name, None)
                if chosen is not None:
                    break
            if chosen is not None:
                self._eye.set_animation(self._next_id(), chosen)
        except Exception as e:
            _log(f"eyes({state}) failed: {e}")
        # ambient LED mood rides along with eye state
        mood = self._LED_MOODS.get(state)
        if mood and self.has.get("led"):
            try:
                color, fade = mood
                activity = self._led.LedActivity()
                activity.mainColor = self._color.from_code(getattr(self._ledcc, color, self._ledcc.Blue))
                activity.fadeColor = self._color.from_code(getattr(self._ledcc, color, self._ledcc.Blue))
                activity.fade_time = fade
                self._led.process_activity(self._next_id(), self._led.LedSide.Both, activity)
            except Exception as e:
                _log(f"led mood failed: {e}")

    def eye_color(self, color_name):
        if not self.has.get("eye"):
            return False
        try:
            code = getattr(self._colorcode, color_name, None)
            if code is None:
                return False
            self._eye.set_iris(self._eye.IrisShape.Modern, code, self._eye.EyeSide.Both)
            return True
        except Exception as e:
            _log(f"eye_color failed: {e}")
            return False

    # ------------------------------------------------------------------- led
    def led_color(self, color_name):
        if not self.has.get("led"):
            return False
        try:
            code = getattr(self._ledcc, color_name, None)
            if code is None:
                return False
            activity = self._led.LedActivity()
            activity.mainColor = self._color.from_code(code)
            activity.fade_time = 0
            self._led.process_activity(self._next_id(), self._led.LedSide.Both, activity)
            return True
        except Exception as e:
            _log(f"led_color failed: {e}")
            return False

    # ------------------------------------------------------------------ arms
    def arm_angle(self, angle, speed=40, wait=True):
        if self.actuators_held() or not self.has.get("arm"):
            return False
        try:
            with self._power_lock:
                if self.actuators_held():
                    return False
                rc = self._arm.set_angle(self._next_id(), self._arm.ArmSide.Both,
                                         speed=speed, angle=angle, with_brake=False)
            if rc < 0:
                return False
            if wait:
                deadline = time.time() + 4
                while time.time() < deadline:
                    if self._arm.get_state(self._arm.ArmSide.Both) == self._arm.ArmState.Completed:
                        break
                    time.sleep(0.05)
            return True
        except Exception as e:
            _log(f"arm_angle failed: {e}")
            return False

    def arms_up(self):
        return self.arm_angle(140)

    def arms_down(self):
        return self.arm_angle(20)

    def fist_bump(self):
        if self.anim and self.anim.play("fist_bump", blocking=True):
            return True
        self.arms_up()
        time.sleep(0.3)
        self.arm_angle(90)
        return True

    def high_five(self):
        if self.anim and self.anim.play("high_five", blocking=True):
            return True
        self.arms_up()
        return True

    # ----------------------------------------------------------------- drive
    def drive_distance(self, mm, speed=45):
        if not self.has.get("drive"):
            return False
        direction = "forward" if mm >= 0 else "backward"
        if self._hazard_active(direction) and not (self._leaving_home or self._escaping):
            _log(f"drive_distance refused: edge hazard latched ({direction})")
            return False
        mobile = self.ensure_mobility()
        if not mobile or not self._motion_allowed(direction):
            _log("drive blocked: docked or edge")
            return False
        try:
            # SDK distance is unsigned; direction is the to_forward flag
            with self._power_lock:
                if (not self._motion_allowed(direction)
                        or (self._hazard_active(direction) and not self._escaping)):
                    return False
                self._drive.go_distance(self._next_id(), int(round(abs(mm))),
                                        int(round(speed)), mm >= 0, True)
            self._watch_motion("forward" if mm >= 0 else "backward")
            self._pose_update(dist_mm=mm)
            return True
        except Exception as e:
            _log(f"drive_distance failed: {e}")
            return False

    def drive_rotate(self, degrees, speed=45):
        if not self.has.get("drive"):
            return False
        if self.actuators_held():
            return False
        if not self._motion_allowed("rotate"):
            return False
        try:
            with self._power_lock:
                if (not self._motion_allowed("rotate")
                        or (self._hazard_active("forward") and not self._escaping)):
                    return False
                self._drive.go_rotate(self._next_id(), int(round(degrees)), False,
                                      int(round(speed)), True, True)
            self._watch_motion("rotate")
            self._pose_update(rot_deg=degrees)
            return True
        except Exception as e:
            _log(f"drive_rotate failed: {e}")
            return False

    def drive_stop(self):
        with self._power_lock:
            if not self.has.get("drive"):
                return False
            ok = True
            # attempt each wheel independently — one failure must not skip the other
            for is_left in (False, True):
                try:
                    self._drive.free_drive(0, is_left, True)
                except Exception as e:
                    _log(f"drive_stop({'left' if is_left else 'right'}) failed: {e}")
                    ok = False
            return ok


    def stop_everything(self):
        """Emergency stop for the STOP command: animations, homing, wheels."""
        self._homing = False
        self._leaving_home = False
        if self.anim:
            self.anim.stop()
        if self.has.get("arm"):
            try:
                self._arm.abort(self._arm.ArmSide.Both)
            except Exception as e:
                _log(f"arm stop failed: {e}")
        return self.drive_stop()

    # --------------------------------------------------------------- homing
    def _pose_update(self, dist_mm=0.0, rot_deg=0.0):
        if self._pose is None:
            return
        import math
        if dist_mm:
            rad = math.radians(self._pose[2])
            self._pose[0] += dist_mm * math.cos(rad)
            self._pose[1] += dist_mm * math.sin(rad)
        if rot_deg:
            self._pose[2] = (self._pose[2] + rot_deg + 180) % 360 - 180

    def _wait_drive_idle(self, timeout=15.0, require_running=False):
        """True when the drive finished. require_running (escape
        verification): only an observed Running -> Completed counts —
        comms errors, controller Error, never-started commands and stale
        home-arrival flags all return False."""
        deadline = time.time() + timeout
        start = time.time()
        saw_running = False
        DS = self._drive.DriveState
        while time.time() < deadline:
            if self._homing and self._home_arrived:
                return True
            try:
                st = self._drive.get_state()
            except Exception:
                return not require_running  # comms error is NOT success
            if not require_running:
                if st != DS.Running:
                    return True
            elif st == DS.Running:
                saw_running = True
            elif st == DS.Completed and saw_running:
                return True
            elif st == DS.Error:
                return False
            elif not saw_running and time.time() - start > 1.0:
                return False  # never started
            time.sleep(0.05)
        return False

    def _undock(self, then_mm=0, speed=25):
        """No autonomous departure until the dock corridor is localized."""
        _log("undock unavailable: safe dock departure is not calibrated")
        return False

    def go_home(self):
        """Charge in place or request manual placement.
        Edge sensors and odometry cannot distinguish the dock lip from a
        table edge, so neither can authorize a blind docking approach."""
        if self.is_on_dock():
            return "already"
        _log("home unavailable: safe dock approach is not calibrated")
        return "unknown"

    # ----------------------------------------------------------- anim queue
    def queue_anim(self, name):
        """Sensor callbacks (foreign threads) request; the main loop plays.
        Animations may only run on the main thread — their sounds call
        snd.play directly, which is not GIL-safe alongside Vosk decoding."""
        self._anim_requests.append(name)

    def drain_anims(self):
        while self._anim_requests:
            name = self._anim_requests.popleft()
            if self.anim:
                self.anim.play(name, blocking=True)

    # variant name -> stock animation file (the REAL stock choreography)
    _DANCE_ANIMS = {"fiesta": "salsa", "groove": "workout", "party": "excited"}

    def dance(self, variant=None):
        """Stock dance choreography through the anim engine — music, arms,
        spins, lights, sunglasses eyes. Plate signature = arms-only party;
        otherwise preflight before the show."""
        if len(self._edge_gaps()) >= 3:
            _log("dance: plate mode (arms only)")
            return self.arms_party()
        if not self._motion_allowed("rotate"):
            _log("dance blocked before show: preflight failed")
            return False
        name = self._DANCE_ANIMS.get(variant, "salsa")
        _log(f"dance: {name} (stock animation)")
        if self.anim and self.anim.play(name, blocking=True):
            self._bump_mood(1)
            return True
        _log("stock animation unavailable — arms party fallback")
        return self.arms_party()

    def arms_party(self):
        """Always-safe celebration: lights, music-less boogie, arms only."""
        try:
            for color in ("Magenta", "Cyan", "Yellow"):
                self._led_flash(color)
            self.eyes("speaking")
            for ang in (150, 30, 150, 30, 140, 40, 150, 30):
                self.arm_angle(ang, speed=75)
            self.mood_eyes("HEARTS")
            self._bump_mood(1)
            return True
        except Exception as e:
            _log(f"arms_party: {e}")
            return False

    def blink(self):
        """Quick blink every few seconds = baseline 'alive' signal."""
        import random
        try:
            self.mood_eyes(random.choice(("BLINK", "BLINK", "BLINK_BIG", "BLINK_SLOW")))
        except Exception:
            pass

    _CURIOS = ("LOOK_LEFT", "LOOK_RIGHT", "DISCOVER", "LOOK_UP", "SCAN",
               "SNEEZE", "BLINK_BIG", "SPARKLING")

    def idle_flourish(self):
        """A little life while waiting: a curious glance, sometimes a small
        stretch. NO random loud noises — a cute pet, not an annoying one.
        Runs on the main thread, so a rare short animation is GIL-safe."""
        import random
        try:
            self.mood_eyes(random.choice(self._CURIOS))
            if random.random() < 0.3:
                ang = random.choice((110, 130, 150))
                self.arm_angle(ang, speed=25, wait=False)
                time.sleep(0.4)
                self.arm_angle(30, speed=25, wait=False)
        except Exception as e:
            _log(f"idle_flourish: {e}")

    def is_on_dock(self):
        self.refresh_power()
        return self.docked

    def wander_step(self):
        """Pet-like exploration: ONE safe move + a curious look.
        Short, preflighted, edge-gated, battery-aware."""
        import random
        try:
            if self.actuators_held():
                self.blink()
                return False
            pct = self.battery_pct()
            low = self.cfg.get("idle", {}).get("low_battery_pct", 10)
            if pct is not None and pct < low and not self.is_on_dock():
                _log(f"wander: battery {pct}% < {low}% — checking return to charger")
                result = self.go_home()
                if result == "unknown":
                    self.speak(f"Battery at {pct} percent. Please carry me to my dock to charge.")
                elif result == "lost":
                    self.speak("I can't find my dock — a little help, please?")
                return True
            if pct is not None and pct < low + 10:
                # low but not critical: conserve, don't roam further
                return False
            if self.is_on_dock():
                # on the dock: eyes and arms only, never wheels
                if random.random() < 0.5:
                    self.idle_flourish()
                else:
                    self.arm_angle(120, speed=45)
                    time.sleep(0.3)
                    self.arm_angle(20, speed=45)
                    self.idle_flourish()
                return True
            if not self._motion_allowed("rotate"):
                return self._escape_edge()
            # roam radius: dead-reckoning drifts, so straying too far from the
            # dock means it's time to head home while home is still findable
            if self._pose is not None:
                import math
                radius = self.cfg.get("idle", {}).get("roam_radius_mm", 700)
                if math.hypot(self._pose[0], self._pose[1]) > radius:
                    _log("wander: roam radius reached — heading home")
                    self.go_home()
                    return True
            move = random.choice(["look", "turn", "scoot", "turn", "scoot"])
            if move == "look":
                self.idle_flourish()
                return True
            if move == "turn":
                self.drive_rotate(random.choice([-90, -60, 60, 90]), speed=35)
            elif move == "scoot":
                self.drive_guarded(random.choice([60, 90, 120]), speed=25)
            self.idle_flourish()
            return True
        except Exception as e:
            _log(f"wander_step: {e}")
            return False

    def _escape_edge(self):
        """Stuck facing a REAL cliff: back up slowly (rear preflight +
        watchdog), then turn away from the gap side. Fires ONLY on a
        confirmed current front gap — dock sensor noise, gap locks, and
        unrelated motion interlocks all return False instead of moving."""
        if self.is_on_dock():
            _log("escape_edge: docked — plate sensor noise is not a cliff")
            return False
        gaps = self._edge_gaps()
        _log(f"escape_edge: gaps={gaps}")
        if len(gaps) >= 4:
            _log("escape_edge: all-four void = airborne — refusing to move")
            return False
        front = [g for g in gaps if g.startswith("Front")]
        if not front:
            _log("escape_edge: no current front gap — staying put")
            return False
        if any(g.startswith("Back") for g in gaps):
            _log("escape_edge: boxed in (front AND back gaps) — staying put")
            return False
        # The hazard latch stays SET for the whole escape — no window where
        # a blind forward command could slip in. The escaper's own drives
        # run under the _escaping exemption (the checks above PROVED
        # backward is clear). Cleared at the end ONLY if the edge is
        # verifiably gone; re-latched "forward" on any failure.
        lock_left = getattr(self, "_gap_lock_until", 0) - time.time()
        if lock_left > 0:
            time.sleep(min(lock_left + 0.2, 4.0))
        self._escaping = True
        try:
            if not self.drive_distance(-80, speed=20):
                self._latch_hazard("forward")  # the front cliff is still there
                _log("escape_edge: backward blocked too — staying put")
                return False
            if not self._wait_drive_idle(timeout=8, require_running=True):
                self.drive_stop()
                self._latch_hazard("forward")
                _log("escape_edge: retreat did not finish — staying put")
                return False
            self.drive_stop()
            still = [g for g in self._edge_gaps() if g.startswith("Front")]
            if still:
                self._latch_hazard("forward")
                _log(f"escape_edge: front gap persists after retreat ({still}) — not rotating")
                return False
            left = any(g.startswith("Front_Left") or g.endswith("Left") for g in gaps)
            right = any(g.startswith("Front_Right") or g.endswith("Right") for g in gaps)
            if left and not right:
                turn = -100  # gap on the left: swing right
            elif right and not left:
                turn = 100
            else:
                turn = 130  # full-front or unknown: big turn either way
            if not self.drive_rotate(turn, speed=25):
                self._latch_hazard("forward")
                _log("escape_edge: rotation rejected — escape FAILED, staying put")
                return False
            if not self._wait_drive_idle(timeout=10, require_running=True):
                self.drive_stop()
                self._latch_hazard("forward")
                _log("escape_edge: turn did not finish — staying put")
                return False
            self.drive_stop()
        finally:
            self._escaping = False
        # clear ONLY if nothing fresh latched during the escape drives:
        # sensor events race this path, so the generation counter decides
        gen0 = getattr(self, "_hazard_gen", 0)
        front_after = [g for g in self._edge_gaps() if g.startswith("Front")]
        if front_after:
            self._latch_hazard("forward")
            _log(f"escape_edge: cliff still ahead after turn {front_after} — hazard kept")
            return False
        with self._hazard_lock:
            if self._hazard_gen == gen0:
                self._edge_hazard = None
                self._hazard_clear_at = None
                self._hazard_airborne = False
                _log("escape_edge: escaped — hazard latch cleared")
                return True
        _log("escape_edge: fresh hazard arrived during escape — latch kept")
        return False

    def _bump_mood(self, delta):
        score = getattr(self, "_mood_score", 3) + delta
        self._mood_score = max(0, min(6, score))

    @property
    def mood(self):
        score = getattr(self, "_mood_score", 3)
        if score >= 5:
            return "ecstatic"
        if score >= 3:
            return "playful"
        if score >= 2:
            return "content"
        return "sleepy"


    # --------------------------------------------------------------- battery
    def battery_pct(self):
        if not self.has.get("battery"):
            return None
        try:
            return int(self._battery.get_capacity())
        except Exception:
            return None

    # ---------------------------------------------------------------- camera
    def take_photo(self, path="/home/pi/spark-photos"):
        try:
            import cv2
            import doly_camera
            if not self.has.get("camera_ok", True):
                return None
            os.makedirs(path, exist_ok=True)
            cam = doly_camera.PiCamera()
            cam.options.photo_width = 1920
            cam.options.photo_height = 1440
            cam.options.verbose = False
            if not cam.start_photo():
                return None
            try:
                frame = cam.capture_photo()
            finally:
                cam.stop_photo()
            if frame is None:
                return None
            out = os.path.join(path, time.strftime("spark_%Y%m%d_%H%M%S.jpg"))
            if not cv2.imwrite(out, frame):
                return None
            return out
        except Exception as e:
            _log(f"take_photo failed: {e}")
            return None

    # ----------------------------------------------------------------- sleep
    def sleep_pose(self):
        if self.anim and self.anim.play("sleep", blocking=True):
            return
        self.eyes("sleepy")
        self.arms_down()
        self.led_color("Black")

    def wake_up(self):
        if self.anim and self.anim.play("wakeup", blocking=True):
            return
        self.eyes("idle")

    # --------------------------------------------------------------- cleanup
    def dispose(self):
        self._power_stop.set()
        if self._power_thread:
            self._power_thread.join(timeout=2)
        self.stop_everything()
        if self._charging:
            try:
                if not self._charging.close():
                    _log("power reader still busy; continuing hardware cleanup")
            except Exception as e:
                _log(f"power reader cleanup failed: {e}")
        for name, mod in (("tts", "_tts"), ("sound", "_snd"), ("eye", "_eye"),
                          ("touch", "_touch"), ("arm", "_arm"), ("led", "_led"),
                          ("battery", "_battery")):
            if self.has.get(name):
                try:
                    getattr(self, mod).dispose()
                except Exception:
                    pass
        if self.has.get("drive"):
            try:
                self._drive.dispose(True)  # dispose IMU as well (SDK contract)
            except TypeError:
                try:
                    self._drive.dispose()
                except Exception:
                    pass
            except Exception:
                pass
        time.sleep(0.2)
