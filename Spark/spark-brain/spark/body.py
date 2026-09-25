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
        self._anim_queue_lock = threading.Lock()
        self._pending_pet = None
        self._interaction = None
        self._interaction_armed = False
        self._interaction_contact = threading.Event()
        self._interaction_stop = threading.Event()
        self._approach_stop = threading.Event()
        self._approaching = False
        self._turning = False
        self.motion_stop_factory = None
        self._person_detector = None
        self._tof_snapshot = None
        self._tof_poll_stop = threading.Event()
        self._tof_poll_thread = None
        self._touch_interrupt = False
        self._dance_index = 0
        # homing: software dead-reckoning (doly_drive.get_position is broken
        # in the pybind layer, so spark tracks its own estimate). Anchored
        # to [0,0,0] whenever charging confirms the seated pose; home.py
        # navigates those remembered coordinates on 'go home'.
        self._pose = None          # [x_mm, y_mm, heading_deg] or None = unknown
        self._homing = False
        self._roaming = False
        self._roam_distance_bound = None
        self._imu_yaw = None
        self._imu_updated_at = 0.0
        self._leaving_home = False
        self._departure = None
        self._docking_entry = None
        self._full_charge_since = None
        self._dock_auto_attempted = False
        self._next_auto_departure = 0.0
        self.last_departure_result = None
        self.last_reseat_result = None
        self._dock_charge_seen_at = None
        self._last_charging_at = None   # monotonic of last charging=True sample
        self._home_arrived = False
        self._edge_hazard = None   # None | "forward" | "backward" | "all":
        # latched after a real gap event — no blind drives TOWARD that edge
        # until she escapes it or it is verifiably gone
        self._hazard_clear_at = None
        self._hazard_airborne = False  # an all-void event followed the latch
        self._escaping = False      # _escape_edge drives bypass the hazard gate
        self.docked = False
        self._dock_hold_path = None
        if hw and cfg.get("state_dir"):
            from pathlib import Path
            self._dock_hold_path = Path(cfg["state_dir"]) / "dock-hold"
            try:
                self.docked = self._dock_hold_path.read_text().strip() == "held"
            except FileNotFoundError:
                pass
            except OSError as exc:
                self.docked = True
                _log(f"dock hold state unreadable: {exc}; holding motors")
        self.sleeping = False
        self._dock_discharge_since = None
        self._charge_notice_pending = False
        self._charge_notice_sent = False
        self._dock_clear_since = None
        self._dock_pickup_at = None
        self._dock_native_stopped = False
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
            if "Down" in str(state) and (
                    (self.anim and self.anim.playing() and self.anim.petting is not True)
                    or self._interaction or self._approaching or self._turning
                    or self._homing or self._leaving_home or self._roaming):
                self._touch_interrupt = True
                self.stop_everything()
                return
            if self._touch_interrupt:
                if "Down" not in str(state):
                    self._touch_interrupt = False
                return  # consume release too; don't queue another performance
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
        # Drive initializes the shared IMU first. A later imu.init() returns
        # already-active, so supplying calibration only there leaves zero
        # offsets in use for both heading and native motor control.
        rc, gx, gy, gz, ax, ay, az = self._helper.get_imu_offsets()
        if rc < 0:
            raise RuntimeError("drive IMU offsets unavailable")
        rc = drive.init(gx, gy, gz, ax, ay, az)
        if rc != 0:
            raise RuntimeError(f"drive init rc={rc}")
        drive.on_complete(lambda i: None)
        drive.on_error(lambda i, s, t: _log(f"drive error id={i} side={s} type={t}"))
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
        # Each side can take 50ms. A 50ms pair interval makes the SDK reader
        # immediately reacquire its mutex and starve getSensorsData. Leave
        # time between pairs; the cached snapshot still expires after 300ms.
        if tof.setup_continuous(150, 60) < 0:
            raise RuntimeError("tof setup_continuous failed")
        self._tof = tof
        import spark_tof_native
        def poll():
            while not self._tof_poll_stop.is_set():
                try:
                    samples = spark_tof_native.read_sensors()
                    self._tof_snapshot = (time.monotonic(), samples)
                except Exception as exc:
                    self._tof_snapshot = None
                    _log(f"ToF reader failed: {exc}")
                self._tof_poll_stop.wait(.03)
        self._tof_poll_thread = threading.Thread(target=poll, daemon=True)
        self._tof_poll_thread.start()

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
                self._note_yaw(data.ypr.yaw)
            except Exception:
                pass

        imu.on_gesture(_on_gesture)
        try:
            imu.on_update(_on_update)
        except Exception:
            pass
        self._imu = imu

    def _note_yaw(self, yaw):
        """Track yaw; a hand rotation retires a stale edge-hazard latch.

        The latch records WHICH WAY a cliff was. Turning her by hand (or
        any large uncommanded rotation) voids that knowledge — holding it
        strands a repositioned robot. Commanded drives never clear it.
        """
        now = time.monotonic()
        previous = self._imu_yaw
        self._imu_yaw = yaw
        self._imu_updated_at = now
        try:
            driving = (self._drive is not None
                       and self._drive.get_state() == self._drive.DriveState.Running)
        except Exception:
            driving = False
        if (driving or self._homing or self._roaming or self._leaving_home
                or self._escaping or self._approaching
                or self._docking_entry is not None):
            self._hand_turn = 0.0
            return
        delta = abs((yaw - (previous if previous is not None else yaw) + 180) % 360 - 180)
        if delta > 45:
            return  # sample glitch
        if delta >= 1.0:
            if now - getattr(self, "_hand_turn_at", 0) <= 1.5:
                self._hand_turn = getattr(self, "_hand_turn", 0.0) + delta
            else:
                self._hand_turn = delta
            self._hand_turn_at = now
        elif now - getattr(self, "_hand_turn_at", 0) > 2.0:
            self._hand_turn = 0.0  # slow drift never accumulates
        if getattr(self, "_hand_turn", 0.0) >= 25 and self._edge_hazard is not None:
            with self._hazard_lock:
                if self._edge_hazard is not None:
                    _log(f"hand turn: clearing stale {self._edge_hazard} edge hazard")
                    self._edge_hazard = None
                    self._hazard_clear_at = None
                    self._hazard_airborne = False
            self._hand_turn = 0.0

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
        # eyes only — the alarm beep on every pickup/bump drove Matt
        # up the wall (2026-09-25). Same policy as the desk-bump tiers.
        "ShockHard": ("DAMAGED", None, None),
        "ShockExtreme": ("DESTROYED", None, None),
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
            if self.sleeping:
                return
            if self._interaction:
                if (self._interaction_armed and family == "imu"
                        and kind in ("ShockLight", "ShockMedium")
                        and not self.speaking_recently()):
                    _log(f"{self._interaction}: contact from {kind}")
                    self._interaction_contact.set()
                return
            if self._turning or (self.anim and self.anim.playing()):
                return  # choreography owns expressive effects; edge/power guards stay live
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
            rc = self._eye.set_animation(
                self._next_id(), getattr(self._eye.expressions, expression_name))
            return rc is None or rc >= 0
        except Exception as e:
            _log(f"mood_eyes failed: {e}")
            return False

    def _init_edge(self):
        """ToF edge sensors — the not-driving-off-tables subsystem."""
        import doly_edge as edge
        rc = edge.init()
        if rc < 0:
            raise RuntimeError(f"edge init rc={rc}")

        def _handle_gap(direction):
            dir_name = str(direction).split(".")[-1]
            if self._docking_entry is not None and self._docking_entry.trailing_gap(dir_name):
                return  # final reverse onto known ramp; rear pair still guards travel
            if self._docking_entry is not None:
                # A short pulse must still cancel entry after the GPIO clears.
                self._docking_entry.reason = "edge"
            if (self._leaving_home and self._departure is not None):
                # ANY gap event during the authorized exit — including the
                # all-void tilt/pickup reading as the nose crosses off the
                # base — is judged by the departure's debounced poll, never
                # torn down from this interrupt. A real cliff or pickup
                # persists and cancels within 150ms (wheels stop in the
                # finally block); a tilt blip costs nothing.
                _log(f"gap event during departure dir={dir_name} — polled check will judge")
                return
            if self._leaving_home:
                self._approach_stop.set()  # transient gaps cancel departure too
            if dir_name == "All" and self.docked:
                self._dock_pickup_at = time.monotonic()
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
                self._roam_distance_bound = None
            try:
                self.drive_stop()
            except Exception:
                pass
            if not self.sleeping:
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
                fault = result is None and not (
                    self._leaving_home and self._departure is not None
                    and self._charging.healthy())
                if fault and not self._power_fault:
                    self._power_fault = True
                    _log(f"power hold: shunt={self._charging.average} "
                         f"error={self._charging.error}")
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
        Release after a verified departure or pickup onto supported ground."""
        if self._charging is None:
            return None
        with self._power_lock:
            charging = self._charging.sample()
            now = time.monotonic()
            gaps = self._edge_gaps()
            if charging is True:
                self._dock_charge_seen_at = now
                if self.docked:
                    # Confirmed-docked charging = electrical proof of where
                    # the dock is; dock-face recovery trusts it for 30 min.
                    self._last_charging_at = now
            elif charging is False or not self._charging.healthy():
                self._dock_charge_seen_at = None
            if self.docked and (len(gaps) == 4 or not gaps):
                # Airborne (all voids) OR set down on open ground: the dock
                # face always shows the front sensor pair, so a latched robot
                # with NO gaps has been lifted somewhere flat (a gentle pickup
                # never reads all four voids).
                self._dock_pickup_at = now
            if charging and not self._leaving_home:
                if not self.docked:
                    self.docked = True
                    self._dock_auto_attempted = False
                    # Placing an undocked robot onto the charger can raise
                    # front/all-gap events before current averaging proves
                    # contact. A new, electrically confirmed dock placement
                    # supersedes that old location's hazard. Never do this
                    # while already held: a failed exit must stay latched.
                    if not gaps or set(gaps) == {"Front_Left", "Front_Right"}:
                        with self._hazard_lock:
                            if self._edge_hazard is not None:
                                _log(f"new dock placement: clearing prior {self._edge_hazard} hazard")
                                self._edge_hazard = None
                                self._hazard_clear_at = None
                                self._hazard_airborne = False
                                self._hazard_gen += 1
                    self._save_dock_hold()
                    # Home frame anchored (stock HomeControl.SetHome): the
                    # seated pose is the world origin, heading 0 = facing
                    # away from the dock. Guarded moves dead-reckon from
                    # here so 'go home' can drive remembered coordinates.
                    self._pose = [0.0, 0.0, 0.0]
                    self._roam_distance_bound = None
                    self.stop_everything()
                    _log("charging confirmed: motors parked")
                elif self._pose is None and self._charging.healthy():
                    # Booted while parked on the charger (dock hold restored):
                    # the seated pose is a known fixed reference, so the home
                    # frame can anchor without a transition event. Healthy
                    # (settled) readings only — a noisy sample never anchors.
                    self._pose = [0.0, 0.0, 0.0]
                self._dock_clear_since = None
                if self._dock_pickup_at is not None and now - self._dock_pickup_at > 3:
                    self._dock_pickup_at = None  # reseated on powered dock
            elif self.docked and not self._leaving_home:
                clear = (charging is False and self._charging.average is not None
                         and self._charging.average < -5
                         and self._dock_pickup_at is not None
                         and self.has.get("edge") and not gaps)
                if clear:
                    if self._dock_clear_since is None:
                        self._dock_clear_since = now
                    elif now - self._dock_clear_since >= 5:
                        self.docked = False
                        self._save_dock_hold()
                        self._pose = None
                        self._dock_clear_since = None
                        self._dock_pickup_at = None
                        self._dock_native_stopped = False
                        _log("removed from charger: pickup and supported ground confirmed")
                else:
                    self._dock_clear_since = None
            if self.docked:
                self._enforce_dock_stop()
            # A full cell tapers around zero current and the fuel gauge can
            # dither a percent or two below full without ever reading 100.
            # Arm at roam_full_pct and keep the dwell through dithering;
            # only undocking, discharge or unhealthy telemetry reset it.
            full_threshold = max(90, min(100, int(
                self.cfg.get("idle", {}).get("roam_full_pct", 100))))
            fullish = (self.battery_pct() is not None
                       and self.battery_pct() >= full_threshold)
            if (self.docked and charging is not False and self._charging.healthy()
                    and fullish and not self._leaving_home):
                if self._full_charge_since is None:
                    self._full_charge_since = now
            elif (not self.docked or charging is False
                    or not self._charging.healthy()):
                self._full_charge_since = None
            # The dock latch is a motor interlock, never proof of charging.
            if self.docked and charging is False and not self._leaving_home:
                if self._dock_discharge_since is None:
                    self._dock_discharge_since = now
                if now - self._dock_discharge_since >= 15 and not self._charge_notice_sent:
                    self._charge_notice_pending = True
                    self._charge_notice_sent = True
                    _log("charger contact lost: sustained discharge; motors remain parked")
            elif charging is True or not self.docked:
                self._dock_discharge_since = None
                self._charge_notice_pending = self._charge_notice_sent = False
            if now >= getattr(self, "_next_power_log", 0):
                self._next_power_log = now + 30
                _log(f"power: battery={self.battery_pct()}% charging={charging} "
                     f"parked={self.docked} shunt={self._charging.average} "
                     f"voltage={self._charging.voltage} error={self._charging.error} "
                     f"gaps={gaps} clear_since={self._dock_clear_since} "
                     f"motors={self._motor_status()}")
            return charging

    def _enforce_dock_stop(self):
        """Check the native controller as well as blocking future requests."""
        departing = (self._leaving_home and self._departure is not None
                     and self._departure.check() is None)
        if self._leaving_home and not departing:
            self._approach_stop.set()
            self._leaving_home = False
            self._dock_native_stopped = False
        if self.has.get("drive") and not departing:
            running = self._drive.get_state() == self._drive.DriveState.Running
            if running or not self._dock_native_stopped:
                _log(f"dock motor stop: native_running={running}")
                self._dock_native_stopped = self.drive_stop()
        if self.has.get("arm"):
            for side in (self._arm.ArmSide.Left, self._arm.ArmSide.Right):
                if self._arm.get_state(side) == self._arm.ArmState.Running:
                    _log(f"dock arm stop: {side}")
                    self._arm.abort(side)

    def _motor_status(self):
        try:
            state = str(self._drive.get_state()) if self.has.get("drive") else "unavailable"
            rpm = [round(float(self._drive.get_rpm(side)), 2) for side in (True, False)] if self.has.get("drive") else []
            arms = [str(self._arm.get_state(side)) for side in (self._arm.ArmSide.Left, self._arm.ArmSide.Right)] if self.has.get("arm") else []
            return {"drive": state, "rpm": rpm, "arms": arms}
        except Exception as exc:
            return {"error": str(exc)}

    def dock_probe(self):
        """Read charging telemetry; never move wheels to test the dock."""
        self.refresh_power()
        return self.docked

    def _save_dock_hold(self):
        """A restart must not forget a charger hold during lost contact."""
        if self._dock_hold_path is not None:
            try:
                temporary = self._dock_hold_path.with_suffix(".tmp")
                temporary.write_text("held" if self.docked else "clear")
                temporary.replace(self._dock_hold_path)
            except OSError as exc:
                _log(f"dock hold persistence failed: {exc}")

    def take_charge_notice(self):
        """One spoken notice per loss of contact, delivered on the main thread."""
        with self._power_lock:
            pending = self._charge_notice_pending
            self._charge_notice_pending = False
            return pending

    def actuators_held(self):
        """Keep arms and wheels still on charge or with uncertain power."""
        if self.sleeping or self._docking_entry is not None:
            return True
        if not self.hw:
            return False
        charging = self.refresh_power()
        pct = self.battery_pct()
        if pct is None or pct <= 2 or charging is None or self._power_fault:
            return True
        # A full battery can taper to zero current. Ambiguous support must
        # also HOLD motion, never serve as permission to cross a plate lip.
        gaps = set(self._edge_gaps())
        # Only the dedicated, bounded side-escape may move with two gaps;
        # all other actions remain immobilized. No bypass for diagonals.
        recoverable_pair = gaps in ({"Front_Left", "Back_Left"},
                                    {"Front_Right", "Back_Right"},
                                    {"Front_Left", "Front_Right"},
                                    {"Back_Left", "Back_Right"})
        escape_owner = (self._escaping and getattr(self, "_escape_thread_id", None)
                        == threading.get_ident())
        if not self.has.get("edge") or (len(gaps) >= 2 and not (escape_owner and recoverable_pair)):
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
            # SDK EdgeControl.h and DetermineGapEvent use LOW for no ground.
            # This is a hardware contract, not a configurable sensitivity.
            # Reversing it locks a supported robot and misses real cliffs.
            state = self._edge.GpioState.Low
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
        rotate -> either end's gaps block the wheel sweep.
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
                  (back and direction in ("backward", "rotate"))
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

    def drive_guarded(self, mm, speed=25, segment_mm=60, interlock=None):
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
            reason = interlock() if interlock else None
            if reason:
                self.drive_stop()
                return reason
            step = min(segment_mm, remaining)
            if not self._motion_allowed(direction):
                return "stopped_edge" if remaining < total else False
            try:
                # pybind11 SupportsInt rejects floats — step/speed are
                # floats after the clamp math above and must be reified
                with self._power_lock:
                    if interlock and self._approach_stop.is_set():
                        return "cancelled"
                    if (not self._motion_allowed(direction)
                            or (self._hazard_active(direction) and not self._escaping)):
                        return False
                    self._drive.go_distance(self._next_id(), int(round(step)),
                                            int(round(speed)), sign > 0, True)
                    if self._roam_distance_bound is not None:
                        self._roam_distance_bound += step  # reserve interrupted travel too
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
                reason = interlock() if interlock else None
                if reason:
                    self.drive_stop()
                    return reason
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
            reason = interlock() if interlock else None
            if reason:
                return reason
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
                        gaps_now = set(self._edge_gaps())
                        safe_retreat = (self._escaping
                            and getattr(self, "_escape_generation", None) == self._hazard_gen
                            and ((gaps_now == {"Front_Left", "Front_Right"} and direction == "backward")
                                 or (gaps_now == {"Back_Left", "Back_Right"} and direction == "forward"))
                            and self.refresh_power() is False
                            and not self._power_fault and self._charging.healthy()
                            and (self.battery_pct() or 0) > 3)
                        if not safe_retreat:
                            self.drive_stop()
                            return
                    if self._drive.get_state() != self._drive.DriveState.Running:
                        return
                    gaps = self._edge_gaps()
                    if gaps:
                        front = any(g.startswith("Front") for g in gaps)
                        back = any(g.startswith("Back") for g in gaps)
                        danger = (front and direction in ("forward", "rotate")) or \
                                 (back and direction in ("backward", "rotate"))
                        if danger:
                            self._gap_lock_until = max(self._gap_lock_until, time.time() + 3.0)
                            if not self.is_on_dock():
                                # a rotate-stop found the cliff with a swept
                                # wheel: latch forward so _escape_edge keeps
                                # the backward retreat route open
                                self._latch_hazard(direction if direction in ("forward", "backward") else
                                                   "all" if front and back else "forward" if front else "backward")
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
                        return True
                    if self._arm.get_state(self._arm.ArmSide.Both) == self._arm.ArmState.Error:
                        return False
                    time.sleep(0.05)
                _log("arm movement did not complete within four seconds")
                return False
            return True
        except Exception as e:
            _log(f"arm_angle failed: {e}")
            return False

    def arms_up(self):
        return self.arm_angle(140)

    def arms_down(self):
        return self.arm_angle(20)

    def fist_bump(self):
        return self._hand_gesture("fist_bump", "fist_ready", 90)

    def high_five(self):
        return self._hand_gesture("high_five", "high_five_ready", 140)

    def _hand_gesture(self, name, ready, angle):
        """Wait for a fresh bump after the arm and ready sound have settled."""
        if self.actuators_held():
            return False
        self._interaction_stop.clear()
        self._interaction_armed = False
        self._interaction = name
        try:
            self.drive_stop()
            if not self.arm_angle(angle, speed=40):
                return False
            if self._interaction_stop.is_set():
                return False
            if self.anim and not self.anim.play(ready):
                return False
            if self._interaction_stop.wait(0.35):
                return False
            self._interaction_contact.clear()
            self._interaction_armed = True
            _log(f"{name}: waiting for contact")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self._interaction_stop.is_set() or self.actuators_held():
                    return False
                if self._interaction_contact.wait(0.02):
                    self._interaction_armed = False
                    return bool(self.anim and self.anim.play(name))
            _log(f"{name}: no contact; lowering arms")
            return True  # offered the gesture; no fake contact celebration
        finally:
            self._interaction_armed = False
            self._interaction = None
            if not self._interaction_stop.is_set():
                self.arm_angle(20, speed=40)

    # ----------------------------------------------------------------- drive
    def drive_distance(self, mm, speed=45):
        if not self._escaping:
            self._roam_distance_bound = None  # ordinary travel loses the visual anchor
        elif self._roam_distance_bound is not None:
            self._roam_distance_bound += abs(mm)
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
                rc = self._drive.go_distance(self._next_id(), int(round(abs(mm))),
                                            int(round(speed)), mm >= 0, True)
                if rc is False or (rc is not None and rc < 0):
                    _log(f"drive rejected: rc={rc}")
                    return False
            self._watch_motion("forward" if mm >= 0 else "backward")
            self._pose_update(dist_mm=mm)
            return True
        except Exception as e:
            _log(f"drive_distance failed: {e}")
            return False

    def drive_rotate(self, degrees, speed=45, from_center=False):
        if not self.has.get("drive"):
            return False
        if self.actuators_held():
            return False
        if not self._motion_allowed("rotate"):
            return False
        try:
            with self._power_lock:
                if (not self._motion_allowed("rotate")
                        or (self._approaching and self._approach_stop.is_set())
                        or ((self._hazard_active("forward") or self._hazard_active("backward"))
                            and not self._escaping)):
                    return False
                rc = self._drive.go_rotate(self._next_id(), int(round(degrees)), from_center,
                                          int(round(speed)), True, True)
                if not from_center:
                    self._roam_distance_bound = None
                elif self._roam_distance_bound is not None:
                    self._roam_distance_bound += 10  # allow for pivot/slip error
                if rc is False or (rc is not None and rc < 0):
                    _log(f"rotate rejected: rc={rc}")
                    return False
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
            # Zeroing PWM alone does not cancel an autonomous target. Abort
            # the native operation first so it cannot reassert wheel speed.
            try:
                self._drive.abort()
            except Exception as e:
                _log(f"drive abort failed: {e}")
                ok = False
            # attempt each wheel independently — one failure must not skip the other
            for is_left in (False, True):
                try:
                    if self._drive.free_drive(0, is_left, True) is False:
                        _log(f"drive_stop({'left' if is_left else 'right'}) rejected")
                        ok = False
                except Exception as e:
                    _log(f"drive_stop({'left' if is_left else 'right'}) failed: {e}")
                    ok = False
            return ok

    def turn_guarded(self, degrees):
        """Finish or stop a voice turn before opening the next listening window."""
        if not self.has.get("drive") or self.actuators_held():
            return "blocked"
        self._approach_stop.clear()
        self._turning = True
        started = time.monotonic()
        seen_running = False
        result = "timeout"
        try:
            stop = self.motion_stop_factory() if self.motion_stop_factory else lambda: False
            if not self.drive_rotate(degrees, speed=30, from_center=True):
                result = "blocked"
                return "blocked"
            while time.monotonic() - started < 15:
                if self._approach_stop.is_set() or stop():
                    result = "cancelled"
                    break
                if self.actuators_held():
                    result = "power"
                    break
                if not self._motion_allowed("rotate"):
                    result = "edge"
                    break
                state = self._drive.get_state()
                if state == self._drive.DriveState.Running:
                    seen_running = True
                elif state == self._drive.DriveState.Error:
                    result = "error"
                    break
                elif state == self._drive.DriveState.Completed and seen_running:
                    result = "ok"
                    break
                elif not seen_running and time.monotonic() - started > 1:
                    result = "not_started"
                    break
                time.sleep(.03)
            return result
        finally:
            self.drive_stop()
            self._turning = False
            _log(f"voice turn: {degrees} degrees result={result}")


    def stop_everything(self):
        """Emergency stop for the STOP command: animations, homing, wheels."""
        self._homing = False
        self._roaming = False
        self._leaving_home = False
        self._interaction_stop.set()
        self._approach_stop.set()
        if self.anim:
            self.anim.stop()
        if self.has.get("arm"):
            try:
                self._arm.abort(self._arm.ArmSide.Both)
            except Exception as e:
                _log(f"arm stop failed: {e}")
        return self.drive_stop()

    def rotate_guarded(self, degrees, interlock):
        """Small center turn, with the approach's stop/proximity checks inline."""
        reason = interlock()
        if reason:
            return reason
        if not self.drive_rotate(max(-30, min(30, degrees)), speed=20, from_center=True):
            return "blocked"
        seen_running = False
        started = time.monotonic()
        try:
            while time.monotonic()-started < 6:
                reason = interlock()
                if reason:
                    return reason
                state = self._drive.get_state()
                if state == self._drive.DriveState.Running:
                    seen_running = True
                elif state == self._drive.DriveState.Completed and seen_running:
                    return "ok"
                elif state == self._drive.DriveState.Error:
                    return "blocked"
                elif not seen_running and time.monotonic()-started > 1:
                    return "blocked"
                time.sleep(.03)
            return "blocked"
        finally:
            self.drive_stop()

    def approach_proximity(self):
        """Both short-range sensors must be healthy and the path clear."""
        if not self.has.get("tof"):
            return "sensor"
        try:
            now = time.monotonic()
            if self.hw:
                snapshot = self._tof_snapshot
                if snapshot is None or now-snapshot[0] > .3:
                    return "sensor"
                details = snapshot[1]
            else:
                samples = self._tof.get_sensors_data()
                details = [(str(s.side), int(s.range_mm), int(s.error), int(s.update_ms)) for s in samples]
            if now >= getattr(self, "_next_approach_sensor_log", 0):
                _log(f"approach proximity: {details}")
                self._next_approach_sensor_log = now + 2
            if len(details) != 2 or len({s[0] for s in details}) != 2:
                _log(f"approach sensor stop: missing pair {details}")
                return "sensor"
            previous = getattr(self, "_approach_sensor_stamps", {})
            for side, distance, error, stamp in details:
                last_stamp, changed = previous.get(side, (None, now))
                if stamp != last_stamp:
                    changed = now
                previous[side] = (stamp, changed)
                if stamp <= 0 or now-changed > .3:
                    _log(f"approach sensor stop: stale {side} age={now-changed:.3f}s {details}")
                    return "sensor"
                # ST DT0020 / UM1983: ECE / max convergence (6/7), max SNR
                # (8) and raw alt (9) report no target; range overflow (13/15)
                # means beyond range, not a close obstacle. Crosstalk/sigma
                # range-ignore (10/11) WITH a -1 SDK range is a rejected
                # measurement, not a contact — glossy tables at grazing
                # angles trip it constantly. A genuinely close object reads
                # underflow (12/14) or a real distance, which still stop.
                if error in (6, 7, 13, 15):
                    continue
                if error in (8, 9, 10, 11) and distance < 0:
                    continue
                if error != 0 or distance < 0:
                    _log(f"approach sensor stop: invalid range/status {details}")
                    return "sensor"
                if distance <= 120:
                    return "obstacle"
            self._approach_sensor_stamps = previous
            return None
        except Exception as exc:
            _log(f"approach proximity failed: {exc}")
            return "sensor"

    def come_here(self):
        """Find one visible person and approach in short, guarded steps."""
        if self._approaching:
            return "busy"
        if not self.hw or not self.has.get("drive"):
            return "unavailable"
        if self.actuators_held():
            if self.docked:
                return "docked"
            # two+ gaps hold the motors: that is an edge situation, not a
            # charging one — never blame the battery for a cliff.
            return "edge" if (self.has.get("edge") and self._edge_gaps()) else "power"
        if self.battery_pct() is None or self.battery_pct() <= 10:
            return "power"
        from .approach import Approach
        from .person_vision import PersonCamera, PersonDetector
        self._approach_stop.clear()
        self._approach_sensor_stamps = {}
        self._approaching = True
        try:
            self.drive_stop()  # finish any preceding manual turn before looking
            stop_check = self.motion_stop_factory() if self.motion_stop_factory else None
            if self._person_detector is None:
                model = self.cfg.get("approach", {}).get("model_path", "/opt/spark/models/nanodet.onnx")
                self._person_detector = PersonDetector(model)
            with PersonCamera(self._person_detector) as camera:
                result = Approach(self, camera, stop_check).run()
            _log(f"come here result={result}")
            return result
        except Exception as exc:
            _log(f"come here failed: {exc}")
            return "unavailable"
        finally:
            self.drive_stop()
            self._approaching = False

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

    def _undock(self):
        """Explicit departure; ordinary reactions never call this method.

        The stock front gap profile gets one measured 20mm forward exit
        only with recent electrical proof of charging. No repeated probes.
        """
        from .departure import Departure
        result = "blocked"
        with self._power_lock:
            self.refresh_power()
            pct = self.battery_pct()
            # A failed voice exit BELOW the automatic charge threshold must
            # not consume the future full-charge roam. If already full,
            # suppress an immediate automatic retry after a rejected command.
            full_threshold = max(90, min(100, int(
                self.cfg.get("idle", {}).get("roam_full_pct", 100))))
            if pct is not None and pct >= full_threshold:
                self._dock_auto_attempted = True
            if (not self.docked or self.sleeping or self._leaving_home
                    or not self.has.get("drive") or not self.has.get("edge")
                    or pct is None or pct <= 2 or not self._charging.healthy()
                    or time.time() < getattr(self, "_gap_lock_until", 0)):
                _log(f"dock departure blocked: battery={pct} sleeping={self.sleeping} "
                     f"parked={self.docked} healthy={self._charging.healthy() if self._charging else False} "
                     f"gap_lock={max(0, getattr(self, '_gap_lock_until', 0)-time.time()):.2f}s")
                self.last_departure_result = result
                return False
            gaps = self._edge_gaps()
            front = any(g.startswith("Front") for g in gaps)
            back = any(g.startswith("Back") for g in gaps)
            direction = "forward"
            known = {"Front_Left", "Front_Right", "Back_Left", "Back_Right"}
            front_probe = (set(gaps) == {"Front_Left", "Front_Right"}
                           and self._dock_charge_seen_at is not None
                           and time.monotonic() - self._dock_charge_seen_at <= 5)
            if (not set(gaps) <= known or (front and back)
                    or (front and not front_probe)
                    or self._hazard_active(direction)):
                _log(f"dock departure blocked: gaps={gaps} front_probe={front_probe} "
                     f"hazard={self._edge_hazard}")
                self.last_departure_result = "edge"
                return False
            self.stop_everything()
            self._approach_stop.clear()
            # Departure stops only on an explicit 'stop': the wake word that
            # authorized this command must not cancel it half a step in.
            stop = self.motion_stop_factory(False) if self.motion_stop_factory else lambda: False
            departure = Departure(self, True, gaps, stop, front_probe=front_probe)
            self._departure = departure
            self._leaving_home = True
            self._dock_native_stopped = False
        _log(f"dock departure: direction={direction} initial_gaps={gaps}")
        try:
            result = departure.run()
            with self._power_lock:
                # Recheck under the same lock used by sensor callbacks and
                # the power monitor before releasing any ordinary actuator.
                if (result == "ok" and departure.check() is None
                        and not self._edge_gaps() and self._charging.charging is False):
                    self.docked = False
                    self._save_dock_hold()
                    self._dock_clear_since = self._dock_pickup_at = None
                    self._next_auto_departure = time.monotonic() + max(
                        60, self.cfg.get("idle", {}).get("roam_cooldown_s", 1800))
                    self._roam_distance_bound = 200  # dock body footprint plus measured exit
                    return True
                result = departure.reason or ("unverified" if result == "ok" else result)
                return False
        except Exception as exc:
            result = "error"
            _log(f"dock departure failed: {exc}")
            return False
        finally:
            with self._power_lock:
                self.drive_stop()
                self._leaving_home = False
                self._departure = None
                self._dock_native_stopped = False
                self.last_departure_result = result
                self.refresh_power()
                if front_probe and result != "ok" and self._edge_gaps():
                    gaps = self._edge_gaps()
                    front = any(g.startswith("Front") for g in gaps)
                    back = any(g.startswith("Back") for g in gaps)
                    self._latch_hazard("all" if front and back else
                                       "forward" if front else "backward")
            _log(f"dock departure: result={result} parked={self.docked}")

    def reseat_probe(self, force=False):
        """Dock profile (front-front gaps) with no electrical contact: one
        slow guarded reverse to seat the pins.

        The identical sensor state facing a real cliff makes the same
        reverse a retreat, so the direction is safe under both readings.
        Rear-edge guards stay armed; charge onset stops the move early.
        """
        # Reset on EVERY call: a gate-skipped probe must not leave a stale
        # failure feeding the idle escalation in __main__.
        self.last_reseat_result = None
        if (self.sleeping or self._homing or self._roaming or self._leaving_home
                or self._docking_entry is not None or self._approaching
                or getattr(self, "_reseating", False)):
            return False
        now = time.monotonic()
        if (now < getattr(self, "_next_forced_reseat_probe", 0) if force
                else now < getattr(self, "_next_reseat_probe", 0)):
            return False
        if self.docked or self.refresh_power() is True:
            return True  # already seated
        gaps = set(self._edge_gaps())
        if gaps != {"Front_Left", "Front_Right"}:
            return False  # only the known dock-face profile
        if (not self.has.get("drive") or not self.has.get("edge")
                or time.time() < getattr(self, "_gap_lock_until", 0)
                or self._hazard_active("backward")):
            return False
        pct = self.battery_pct()
        if pct is None or pct <= 2 or not self._charging.healthy():
            return False  # uncertain power stays held
        self._next_reseat_probe = now + 600  # consume the cadence only on a real attempt
        if force:
            self._next_forced_reseat_probe = now + 600
        stop = self.motion_stop_factory(False) if self.motion_stop_factory else (lambda: False)
        self._approach_stop.clear()
        self._reseating = True
        deadline = time.monotonic() + 30
        seated, moved, uncertain = False, False, False

        def _verify_seat():
            settle = time.monotonic() + 6
            while time.monotonic() < settle:
                if self.refresh_power() is True:
                    return True
                time.sleep(.1)
            return False

        _log("reseat probe: dock profile without contact — seating")
        try:
            travelled = 0
            # A slip can leave her more than one probe-length off the pins.
            # Reverse is safe under both readings (seating at the dock,
            # retreating from a cliff lip), so allow up to 75mm with a
            # contact check after every 12mm step.
            while travelled < 75 and time.monotonic() < deadline:
                contact = False
                with self._power_lock:
                    self.refresh_power()
                    if (self._charging.charging is True or stop() or self.sleeping
                            or self._approach_stop.is_set()
                            or any(g.startswith("Back") for g in self._edge_gaps())
                            or self._hazard_active("backward")):
                        break
                    step = min(12, 75 - travelled)
                    rc = self._drive.go_distance(self._next_id(), step, 10, False, True)
                    if rc is False or (rc is not None and rc < 0):
                        break
                moved = True
                # 12mm at speed 10 takes ~1.2s; give the watchdog real margin.
                end, running, complete = time.monotonic() + 2.5, False, False
                while time.monotonic() < end:
                    with self._power_lock:
                        if self.refresh_power() is True:
                            contact = True
                            break
                        if (stop() or self.sleeping or self._approach_stop.is_set()
                                or any(g.startswith("Back") for g in self._edge_gaps())
                                or self._hazard_active("backward")):
                            break
                    state = self._drive.get_state()
                    if state == self._drive.DriveState.Running:
                        running = True
                    elif state == self._drive.DriveState.Completed:
                        # a 12mm step at speed 10 can finish before the first poll
                        complete = True
                        break
                    elif state == self._drive.DriveState.Error:
                        break
                    time.sleep(.03)
                self.drive_stop()
                if contact:
                    seated = _verify_seat()
                    break
                if not complete:
                    uncertain = True  # issued motion with unmeasured travel
                    break
                travelled += step
                if _verify_seat():
                    seated = True
                    break
        finally:
            self.drive_stop()
            self._reseating = False
        if seated:
            _log("reseat probe: charge confirmed — seated")
            self.last_reseat_result = "seated"
        elif moved:
            self.last_reseat_result = "no_contact"
            # An unconfirmed probe is not a pickup or a new anchor, but its
            # real travel still counts. Unmeasured partial travel instead
            # invalidates the frame — a falsely trusted pose is worse
            # than no pose (dock-face recovery keys off it).
            if uncertain:
                self._pose = None
            elif travelled:
                self._pose_update(dist_mm=-travelled)
            if self._roam_distance_bound is not None:
                self._roam_distance_bound += travelled
            _log("reseat probe: no contact after seating move")
            if time.time() >= getattr(self, "_next_reseat_speak", 0):
                self._next_reseat_speak = time.time() + 3600
                self.speak("I'm sitting on my dock, but I'm not charging. Is my dock plugged in?")
        return seated

    def recover_dock_face(self):
        """Failed reseat at the dock face: retreat forward to open ground,
        then run the full visual return — only when the live home frame
        PROVES this ground is hers (anchored at this dock, still within
        600mm of the origin, so she has driven this exact floor recently).

        The front-pair profile alone is ambiguous (dock plate lip or a real
        cliff edge). Three proofs disambiguate it: the pose frame (anchored
        at this dock, still near the origin, heading still near 0 — a slip,
        not wandering), AND electrical recency (CONFIRMED-docked charging
        seen here within the last 30 minutes). Without them the old
        behavior stands: stay held, keep probing, speak hourly.
        """
        import math
        if (self.sleeping or self._homing or self._roaming or self._leaving_home
                or self._docking_entry is not None or self._approaching
                or getattr(self, "_reseating", False)):
            return False
        now = time.monotonic()
        if now < getattr(self, "_next_dock_face_recovery", 0):
            return False
        if self.docked or self.refresh_power() is True:
            return True  # seated by the probe or a nudge since
        if set(self._edge_gaps()) != {"Front_Left", "Front_Right"}:
            return False
        pose = self._pose
        if pose is None or math.hypot(pose[0], pose[1]) > 600:
            return False  # no proof this ground is the dock's front lip
        if abs(pose[2]) > 60:
            return False  # slips leave her near heading 0; a turned pose
                          # means wandering, not a slip — no proof of the lip
        charge_at = getattr(self, "_last_charging_at", None)
        if charge_at is None or now - charge_at > 1800:
            return False  # no recent confirmed-docked charging proof
        if (not self.has.get("drive") or not self.has.get("edge")
                or time.time() < getattr(self, "_gap_lock_until", 0)):
            return False
        pct = self.battery_pct()
        # Aligned with go_home's own <=3 refusal: retreating at 3% only to
        # be refused the visual return would strand her beside the dock.
        if pct is None or pct <= 3 or not self._charging.healthy():
            return False
        _log(f"dock-face recovery: pose={pose} — retreating to open ground")
        DS = self._drive.DriveState
        travelled, uncertain = 0, False
        try:
            # Raw steps like Departure's: the exact, unchanging front-pair
            # profile is tolerated mid-step (it IS why she is pinned); gaps,
            # contact and charging are polled DURING the step and any change
            # stops the motors immediately.
            while travelled < 60 and set(self._edge_gaps()) == {"Front_Left", "Front_Right"}:
                with self._power_lock:
                    self.refresh_power()
                    if (self._charging.charging is True or self.sleeping
                            or self._approach_stop.is_set()):
                        break
                    rc = self._drive.go_distance(self._next_id(), 20, 15, True, True)
                    if rc is False or (rc is not None and rc < 0):
                        break
                end, running, complete = time.monotonic()+2.5, False, False
                while time.monotonic() < end:
                    with self._power_lock:
                        if (self.refresh_power() is True
                                or set(self._edge_gaps()) != {"Front_Left", "Front_Right"}
                                or self.sleeping or self._approach_stop.is_set()):
                            self.drive_stop()
                            uncertain = True  # stopped mid-step: travel unmeasured
                            break
                    state = self._drive.get_state()
                    if state == DS.Running:
                        running = True
                    elif state == DS.Completed and running:
                        complete = True
                        break
                    elif state == DS.Error:
                        uncertain = True
                        break
                    time.sleep(.03)
                self.drive_stop()
                if not complete:
                    # Window expired or ended unconfirmed: the issued step
                    # may have moved — the pose can no longer be trusted.
                    uncertain = True
                    break
                travelled += 20
                self._pose_update(dist_mm=20)
        finally:
            self.drive_stop()
        if uncertain:
            self._pose = None  # a falsely trusted frame is worse than none
        gaps = set(self._edge_gaps())
        if gaps:
            # Failed or interrupted attempt: short backoff, not the long
            # cooldown — a stranded low-battery robot must retry soon.
            self._next_dock_face_recovery = time.monotonic() + 180
            _log(f"dock-face recovery: still pinned after {travelled}mm gaps={gaps}")
            return False
        _log(f"dock-face recovery: clear after {travelled}mm — full visual return")
        self._next_dock_face_recovery = time.monotonic() + 1800
        return self.go_home() in ("arrived", "already")

    def dock_roam_ready(self):
        """One automatic departure per dock visit after a full minute at 100%."""
        ready = (self.cfg.get("idle", {}).get("roam_enabled", True)
                and self.cfg.get("homing", {}).get("enabled", False)
                and self.docked and not self.sleeping and not self._dock_auto_attempted
                and self._full_charge_since is not None
                and time.monotonic() - self._full_charge_since >= 60
                and time.monotonic() >= self._next_auto_departure)
        if not ready:
            return False
        if any(g.startswith("Front") for g in self._edge_gaps()):
            return (self._dock_charge_seen_at is not None
                    and time.monotonic() - self._dock_charge_seen_at <= 5)
        return True

    def go_home(self):
        """Return using the visible dock; only electrical contact is arrival.

        One attempt closes at most ~1200mm of guarded travel, so distance
        needs retries: each attempt sweeps for the marker fresh. Recoverable
        results (limit, lost, alignment, a caught track on entry) get another
        attempt while battery allows; deliberate stops and power faults do not.
        """
        if self.is_on_dock():
            return "already"
        if self._homing:
            return "busy"
        if not self.hw or not self.cfg.get("homing", {}).get("enabled", False):
            return "unknown"
        pct = self.battery_pct()
        if pct is None or pct <= 3:
            return "power"
        self._approach_stop.clear()  # fresh explicit return, including edge recovery
        gaps = set(self._edge_gaps())
        # The seated front-pair profile is also a possible real cliff. The
        # 25mm reverse probe is safe in either case; only electrical contact
        # counts as arrival. Don't wait for the idle probe's 10-minute timer.
        front_pair = gaps == {"Front_Left", "Front_Right"}
        if front_pair and self.reseat_probe(force=True):
            return "arrived"
        if front_pair and not self._edge_gaps():
            # A 25mm probe clearing a REAL cliff is not enough clearance to
            # safely sweep the corners during the camera search. Continue
            # backing up with live rear-gap/power guards before any rotation.
            result = self.drive_guarded(-60, speed=15, segment_mm=20,
                interlock=lambda: "cancelled" if self._approach_stop.is_set() else None)
            if result != "ok":
                return "arrived" if self.is_on_dock() else "edge"
        if self._edge_gaps():
            _log("go_home: recovering from edge before marker search")
            if not self._escape_edge():
                return "edge"
        # An explicitly commanded return outvotes a stale directional latch:
        # live sensors read clear and every homing step re-checks real gaps,
        # so a genuine lip re-stops her immediately. The 60s quiet-hold is
        # for uncommanded wandering, not for answering her owner.
        if not self._edge_gaps() and (self._hazard_active("forward")
                                      or self._hazard_active("backward")):
            with self._hazard_lock:
                _log("go_home: live sensors clear — retiring stale edge latch")
                self._edge_hazard = None
                self._hazard_clear_at = None
                self._hazard_airborne = False
        from .homing import Homing
        self._approach_stop.clear()
        self._approach_sensor_stamps = {}
        self._homing = True
        try:
            stop = self.motion_stop_factory() if self.motion_stop_factory else None
            try:
                attempts = int(self.cfg.get("homing", {}).get("attempts", 3))
            except (TypeError, ValueError):
                attempts = 3
            attempts = max(1, min(4, attempts))
            # A dead or dying pack must not even start the trip home.
            pct = self.battery_pct()
            if pct is not None and pct <= 3:
                return "power"
            recoverable = {"limit", "lost", "not_found", "alignment",
                           "too_close", "turn_unverified", "timeout"}
            final = "sensor"
            for attempt in range(1, attempts + 1):
                result = Homing(self, stop).run()
                _log(f"home attempt {attempt}/{attempts} result={result}")
                final = result
                if result not in recoverable:
                    break
                if self.is_on_dock():
                    return "arrived"
                pct = self.battery_pct()
                if pct is None or pct <= 3:  # unknown telemetry fails safe
                    break
                if attempt < attempts:
                    self.drive_stop()
                    time.sleep(2)  # settle sensors; fresh camera session next
            return final
        except InterruptedError:
            return "cancelled"
        except Exception as exc:
            _log(f"home failed: {exc}")
            return "sensor"
        finally:
            self.drive_stop()
            self._homing = False

    # ----------------------------------------------------------- anim queue
    def queue_anim(self, name):
        """Sensor callbacks (foreign threads) request; the main loop plays.
        Animations may only run on the main thread — their sounds call
        snd.play directly, which is not GIL-safe alongside Vosk decoding."""
        with self._anim_queue_lock:
            if not self.sleeping:
                if name in {"petting1", "petting2", "petting3"}:
                    self._pending_pet = name  # latest level, no long backlog of strokes
                else:
                    self._anim_requests.append(name)

    def petting_active(self):
        return (self._pending_pet is not None or bool(
            self.anim and self.anim.playing() and self.anim.petting is True))

    def drain_anims(self):
        # At most one performance per main-loop pass, so petting cannot
        # starve wake-word capture indefinitely.
        with self._anim_queue_lock:
            if self.sleeping:
                self._anim_requests.clear()
                self._pending_pet = None
                return
            if self._pending_pet is not None:
                name, self._pending_pet = self._pending_pet, None
            elif self._anim_requests:
                name = self._anim_requests.popleft()
            else:
                return
        if self.anim:
            self.anim.play(name, blocking=True)

    # variant name -> stock animation file (the REAL stock choreography)
    _DANCE_ANIMS = {"fiesta": "salsa", "salsa": "salsa", "groove": "workout",
                    "workout": "workout", "party": "excited_1", "twist": "twist",
                    "rock": "rock", "meditate": "meditate", "fireman": "fireman",
                    "policeman": "policeman"}

    def dance(self, variant=None):
        """Stock choreography. Each motor command retains its safety gate;
        docked performances can still use eyes, lights and music."""
        if variant is None:
            variants = ("salsa", "twist", "rock", "party")
            variant = variants[self._dance_index % len(variants)]
            self._dance_index += 1
        name = self._DANCE_ANIMS.get(variant, "salsa")
        _log(f"dance: {name} (stock animation)")
        if self.anim and self.anim.play(name, blocking=True):
            self._bump_mood(1)
            return True
        return False  # never restart a cancelled/failed performance via fallback

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

    def _return_margin_pct(self):
        """Battery reserve for the trip home from the anchored roam distance.

        Far from the dock, return BEFORE the flat low-water mark so the
        remaining charge covers the guarded approach + entry retries.
        """
        import math
        try:
            bound = float(self._roam_distance_bound or 0)
            per_m = float(self.cfg.get("idle", {}).get("roam_reserve_pct_per_m", 2))
        except (TypeError, ValueError):
            return 0  # invalid config/telemetry degrades to the flat threshold
        if bound <= 0 or per_m <= 0 or not math.isfinite(bound) or not math.isfinite(per_m):
            return 0
        return min(10, math.ceil(bound / 1000.0) * per_m)

    def wander_step(self):
        """Pet-like exploration: ONE safe move + a curious look.
        Short, preflighted, edge-gated, battery-aware."""
        try:
            if not self.cfg.get("idle", {}).get("roam_enabled", True):
                return False
            if self.docked:
                if not self.dock_roam_ready():
                    return False
                self._dock_auto_attempted = True
                return self._undock()  # no additional random move this idle pass
            if self.actuators_held():
                self.blink()
                return False
            pct = self.battery_pct()
            low = self.cfg.get("idle", {}).get("low_battery_pct", 10)
            if pct is None or pct <= low + self._return_margin_pct():
                return False  # the central battery check owns return/retry notices
            if not self._motion_allowed("rotate"):
                return self._escape_edge()
            if not self.cfg.get("homing", {}).get("enabled", False):
                return False  # free roaming requires a working return path
            from .roaming import Roaming
            self._approach_stop.clear()
            self._roaming = True
            stop = self.motion_stop_factory() if self.motion_stop_factory else None
            result = Roaming(self, stop).run()
            _log(f"roam result={result} distance_bound={self._roam_distance_bound}")
            return result == "ok"
        except Exception as e:
            _log(f"wander_step: {e}")
            return False
        finally:
            self._roaming = False
            self.drive_stop()

    def _escape_side_edge(self, side):
        """Turn a side straddle inward in short center-pivot pulses.

        A new gap event aborts the native drive immediately; never attempt
        translation until the leading sensors are supported again.
        """
        original = {f"Front_{side}", f"Back_{side}"}
        from .homing import angle_delta
        generation = self._hazard_gen
        self._escaping = True
        self._escape_thread_id = threading.get_ident()
        try:
            for _ in range(5):
                gaps = set(self._edge_gaps())
                if gaps - original or self._hazard_gen != generation:
                    return False
                if not any(g.startswith("Front") for g in gaps):
                    break
                if (not self.has.get("drive") or not self.has.get("edge")
                        or self.sleeping or not self._charging.healthy()
                        or self.battery_pct() is None or self.battery_pct() <= 3
                        or self.refresh_power() is not False):
                    return False
                before = self._imu_yaw
                if before is None or time.monotonic()-self._imu_updated_at > .3:
                    return False
                # Positive SDK turns reduce yaw: right void -> left turn.
                command = -12 if side == "Right" else 12
                with self._power_lock:
                    if set(self._edge_gaps()) - original or self._hazard_gen != generation:
                        return False
                    rc = self._drive.go_rotate(self._next_id(), command, True, 10, True, True)
                    if rc is False or (rc is not None and rc < 0):
                        return False
                until = time.monotonic()+3
                while time.monotonic() < until:
                    if (self._hazard_gen != generation or set(self._edge_gaps()) - original
                            or self._approach_stop.is_set() or self.sleeping):
                        return False
                    state = self._drive.get_state()
                    if state == self._drive.DriveState.Error:
                        return False
                    if state == self._drive.DriveState.Completed:
                        break
                    time.sleep(.03)
                else:
                    return False
                self.drive_stop()
                delta = (angle_delta(self._imu_yaw, before)
                         if self._imu_yaw is not None else 0)
                if (time.monotonic()-self._imu_updated_at > .3
                        or delta * (-command) < 2):
                    return False
                if self._roam_distance_bound is not None:
                    self._roam_distance_bound += 15
            else:
                return False
            # No forward gap: a short guarded move carries the unsupported
            # rear corner inward. A new front gap or callback stops instantly.
            if self._edge_gaps():
                result = self.drive_guarded(40, speed=15, segment_mm=20,
                    interlock=lambda: "edge" if self._hazard_gen != generation
                    or any(g.startswith("Front") for g in self._edge_gaps()) else None)
                if result != "ok":
                    return False
            if self._hazard_gen != generation or self._edge_gaps():
                return False
            with self._hazard_lock:
                if self._hazard_gen == generation:
                    self._edge_hazard = None
                    self._hazard_clear_at = None
                    self._hazard_airborne = False
                    _log("escape_edge: side straddle cleared")
                    return True
            return False
        finally:
            self.drive_stop()
            self._escaping = False
            self._escape_thread_id = None

    def _escape_edge(self):
        """Bounded retreat from an edge; ambiguous profiles remain held."""
        if self.is_on_dock():
            _log("escape_edge: docked — plate sensor noise is not a cliff")
            return False
        gaps = self._edge_gaps()
        _log(f"escape_edge: gaps={gaps}")
        now = time.monotonic()
        history = [t for t in getattr(self, "_edge_escape_attempts", []) if now-t < 180]
        if len(history) >= 3:
            _log("escape_edge: retry budget exhausted — waiting for help")
            return False
        self._edge_escape_attempts = history + [now]
        profile = set(gaps)
        if profile in ({"Front_Left", "Back_Left"},
                       {"Front_Right", "Back_Right"}):
            gen = self._hazard_gen
            lock_left = getattr(self, "_gap_lock_until", 0)-time.time()
            if lock_left > 0:
                time.sleep(min(lock_left+.2, 4))
            if self._hazard_gen != gen or set(self._edge_gaps()) != profile:
                return False
            return self._escape_side_edge("Left" if "Front_Left" in profile else "Right")
        if profile == {"Back_Left", "Back_Right"}:
            # Front is supported: a short forward step gets both rear
            # sensors back over the table. A new gap/event cancels it.
            gen = self._hazard_gen
            lock_left = getattr(self, "_gap_lock_until", 0)-time.time()
            if lock_left > 0:
                time.sleep(min(lock_left+.2, 4))
            if self._hazard_gen != gen or set(self._edge_gaps()) != profile:
                return False
            self._escaping = True
            self._escape_generation = gen
            self._escape_thread_id = threading.get_ident()
            try:
                result = self.drive_guarded(60, speed=15, segment_mm=20,
                    interlock=lambda: "edge" if self._hazard_gen != gen
                    or any(g.startswith("Front") for g in self._edge_gaps()) else None)
                if result != "ok" or self._edge_gaps() or self._hazard_gen != gen:
                    return False
                with self._hazard_lock:
                    if self._hazard_gen != gen:
                        return False
                    self._edge_hazard = None
                    self._hazard_clear_at = None
                    self._hazard_airborne = False
                return True
            finally:
                self.drive_stop()
                self._escaping = False
                self._escape_thread_id = None
                self._escape_generation = None
        if len(profile) >= 3 or (any(g.startswith("Front") for g in gaps)
                                  and any(g.startswith("Back") for g in gaps)):
            _log("escape_edge: mixed/airborne gaps — refusing to move")
            return False
        front = [g for g in gaps if g.startswith("Front")]
        if not front:
            _log("escape_edge: no current front gap — staying put")
            return False
        # The hazard latch stays SET for the whole escape — no window where
        # a blind forward command could slip in. The escaper's own drives
        # run under the _escaping exemption (the checks above PROVED
        # backward is clear). Cleared at the end ONLY if the edge is
        # verifiably gone; re-latched "forward" on any failure.
        gen0 = getattr(self, "_hazard_gen", 0)
        lock_left = getattr(self, "_gap_lock_until", 0) - time.time()
        if lock_left > 0:
            time.sleep(min(lock_left + 0.2, 4.0))
        if self._hazard_gen != gen0 or set(self._edge_gaps()) != profile:
            return False
        self._escaping = True
        self._escape_generation = gen0
        self._escape_thread_id = threading.get_ident()
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
            still = self._edge_gaps()
            if still or self._hazard_gen != gen0:
                self._latch_hazard("forward")
                _log(f"escape_edge: gap/event after retreat ({still}) — not rotating")
                return False
            left = any(g.startswith("Front_Left") or g.endswith("Left") for g in gaps)
            right = any(g.startswith("Front_Right") or g.endswith("Right") for g in gaps)
            if left and not right:
                turn = -100  # gap on the left: swing right
            elif right and not left:
                turn = 100
            else:
                turn = 130  # full-front or unknown: big turn either way
            if not self.drive_rotate(turn, speed=25, from_center=True):
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
            self._escape_thread_id = None
            self._escape_generation = None
        # clear ONLY if nothing fresh latched during the escape drives:
        # sensor events race this path, so the generation counter decides
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
        # Quiet standby, not a cancellable stock snoring performance. Never
        # reposition arms/wheels on the charger to achieve a cosmetic pose.
        self.sleeping = True
        self.react_enabled = False
        self.stop_everything()
        self._anim_requests.clear()
        self._pending_pet = None
        self._sfx_queue.clear()
        self.eyes("sleepy")
        self.led_color("Black")
        _log("sleep: standby; idle motion and follow-ups disabled")

    def wake_up(self):
        self.sleeping = False
        self.react_enabled = True
        self.eyes("idle")
        _log("sleep: awake")

    # --------------------------------------------------------------- cleanup
    def dispose(self):
        self._tof_poll_stop.set()
        if self._tof_poll_thread:
            self._tof_poll_thread.join(timeout=1)
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
