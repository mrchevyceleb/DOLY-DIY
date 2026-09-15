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

TTS_WAV = "/tmp/spark_tts.wav"


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
        "ObjectComing": ("CAUTIOUS", "click", None),   # eyes/sfx only — NEVER auto-drive
        "ObjectGoing": ("HAPPY", None, None),            # (see table-fall postmortem)
        "Scrubing": ("SPARKLING", "pet", None),
        "ToLeft": ("LOOK_LEFT", None, None),
        "ToRight": ("LOOK_RIGHT", None, None),
    }
    _IMU_REACTIONS = {
        "ShockLight": ("BUMP", "click", None),
        "ShockMedium": ("BUGGED", "damage", None),
        "ShockHard": ("DAMAGED", "alarm", None),
        "ShockExtreme": ("DESTROYED", "alarm", None),
        "ShortShake": ("DIZZY_L", "debuff", None),
        "LongShake": ("DIZZY_R", "debuff", None),
        "Vibrate": ("NERVOUS", "click", None),
        "VibrateExtreme": ("FRIGHTENED", "alarm", None),
        "Move": ("LOOK_AHEAD", None, None),
    }

    def _sensor_react(self, family, kind):
        """Thread-safe, debounced stock-style reactions. Never raises."""
        try:
            if not getattr(self, "react_enabled", True):
                return
            now = time.time()
            key = f"{family}:{kind}"
            if now - self._react_debounce.get(key, 0) < 3.0:
                return
            self._react_debounce[key] = now
            table = self._TOF_REACTIONS if family == "tof" else self._IMU_REACTIONS
            expr, sfx, motion = table.get(kind, (None, None, None))
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

        def _on_gap(direction):
            # EMERGENCY: kill motion, lock further motion, react
            dir_name = str(direction).split(".")[-1]
            lock_s = 10.0 if dir_name == "All" else 3.0  # All = airborne/off-edge
            self._gap_lock_until = max(self._gap_lock_until, time.time() + lock_s)
            _log(f"GAP DETECTED dir={dir_name} — motion locked {lock_s}s")
            try:
                self.drive_stop()
            except Exception:
                pass
            self.eyes("thinking")
            self._led_flash("Red")
            if dir_name == "All":
                self.mood_eyes("FRIGHTENED")

        edge.on_gap_detect(_on_gap)
        rc = edge.enable_control()
        if rc < 0:
            raise RuntimeError(f"edge enable_control rc={rc}")
        self._gap_lock_until = 0.0
        self._edge = edge

    # ------------------------------------------------------------------- TTS
    def speak(self, text, wait=True):
        """Say something with her stock voice. Returns False if muted."""
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
                # strip anything the synth would read literally
                text = re.sub(r"[*_`#>]+", "", text)
                self._tts.produce(text)
                self._snd.play(TTS_WAV, self._next_id())  # (file, block_id)
                if wait:
                    dur = self._wav_duration(TTS_WAV)
                    time.sleep(dur + 0.15)
                    # wake-word echo suppression: don't "hear" ourselves
                    self._speaking_until = time.time() + dur + 1.0
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

    def wake_reaction(self):
        """'Hey Spark' acknowledged: stock wake chirp + WAKE_WORD eyes + cyan.

        defer=False: this runs on the main thread (right after the wake
        listener returns), so direct playback is GIL-safe — the deferred
        queue wouldn't flush until next turn and the chirp would be silent.
        """
        chirp = self.cfg.get("wake", {}).get("chirp")
        if chirp:
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
        sentences = list(sentences)
        if not sentences:
            return
        if not (self.has.get("tts") and self.has.get("sound")):
            for sent in sentences:
                self.speak(sent, wait=False)
            return

        import shutil
        prev_end = 0.0
        played = []
        with self._tts_lock:
            for i, sent in enumerate(sentences):
                text = re.sub(r"[*_`#>]+", "", (sent or "").strip())
                if not text:
                    continue
                tmp = f"/tmp/spark_tts_{i}.wav"
                self._tts.produce(text)          # writes TTS_WAV (blocking)
                shutil.copyfile(TTS_WAV, tmp)
                # wait for the previous sentence to finish playing
                now = time.time()
                if prev_end > now:
                    time.sleep(prev_end - now)
                self._snd.play(tmp, self._next_id())
                prev_end = time.time() + self._wav_duration(tmp) + 0.05
                played.append(tmp)
            # let the last sentence finish
            now = time.time()
            if prev_end > now:
                time.sleep(prev_end - now)
            for tmp in played:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        self._speaking_until = prev_end + 1.0

    def pet_pulse(self):
        """Instant 'I felt that' reaction: sfx chirp + LED flash + happy eyes."""
        sfx = self.cfg.get("sounds", {}).get("pet_sfx")
        if sfx:
            self.play_sfx(sfx)
        self._led_flash("Cyan")
        self.eyes("listening")

    # ---------------------------------------------------------- dock sensing
    def dock_probe(self):
        """Truth test: command a small rotate; if the gyro doesn't move,
        the wheels aren't touching anything (charging dock). Result cached;
        re-probes are cheap and safe (dock spin is invisible)."""
        if not (self.has.get("drive") and self.has.get("imu")):
            self.docked = False
            return False
        try:
            yaw_before = getattr(self, "_imu_yaw", None)
            if yaw_before is None:
                time.sleep(0.2)
                yaw_before = getattr(self, "_imu_yaw", 0.0)
            self._drive.go_rotate(self._next_id(), 12, False, 30, True, True)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self._drive.get_state() != self._drive.DriveState.Running:
                    break
                time.sleep(0.05)
            time.sleep(0.3)
            yaw_after = getattr(self, "_imu_yaw", yaw_before)
            delta = abs(yaw_after - yaw_before)
            self.docked = delta < 5.0
            _log(f"dock_probe: yaw delta {delta:.1f} deg -> docked={self.docked}")
            return self.docked
        except Exception as e:
            _log(f"dock_probe failed: {e}")
            self.docked = False
            return False

    def ensure_mobility(self):
        """True if she can actually drive. Re-probes when docked (she may
        have been lifted off)."""
        if not getattr(self, "docked", False):
            return True
        return not self.dock_probe()

    # ------------------------------------------------------------- edge lock
    def _edge_gaps(self):
        """Which sensors currently see a void (['Front_Left', 'Back_Right', ...])."""
        if not self.has.get("edge"):
            return []
        try:
            state = getattr(self._edge.GpioState,
                            self.cfg.get("edge", {}).get("gap_gpio_state", "Low"))
            return [str(getattr(s, "id", "?")).split(".")[-1]
                    for s in self._edge.get_sensors(state)]
        except Exception as e:
            _log(f"preflight poll failed: {e}")
            return []  # poll bug must not brick motion — events still guard

    def _motion_allowed(self, direction="forward"):
        """Direction-aware safety: a cliff BEHIND her must not block forward
        motion (that bug trapped her on the dock and froze her near desk
        edges). forward -> only Front gaps block; backward -> only Back gaps;
        rotate -> Front gaps block (Back-only tolerated; watchdog guards
        mid-rotation sweeps).
        """
        if not self.has.get("edge"):
            return True
        if time.time() < getattr(self, "_gap_lock_until", 0):
            _log(f"motion blocked ({direction}): gap lock active")
            return False
        gaps = self._edge_gaps()
        if not gaps:
            return True
        if direction == "rotate" and len(gaps) >= 4:
            # all-four-void is the dock signature — and the dock probe PROVED
            # in-place pivoting is safe there. Dance away on the charger.
            _log(f"dock pivot allowed (gaps={gaps})")
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

    def _watch_motion(self, direction="forward"):
        """While a drive command runs, poll edges and hard-stop on a gap that
        matters for THIS direction. Mirrors _motion_allowed — including its
        dock-pivot rule (all-four void on a rotate is the dock signature:
        let her spin, killing it here undoes the preflight allowance)."""
        def _run():
            try:
                deadline = time.time() + 15
                while time.time() < deadline:
                    if self._drive.get_state() != self._drive.DriveState.Running:
                        return
                    gaps = self._edge_gaps()
                    if gaps:
                        front = any(g.startswith("Front") for g in gaps)
                        back = any(g.startswith("Back") for g in gaps)
                        dock_spin = direction == "rotate" and len(gaps) >= 4
                        danger = (front and direction in ("forward", "rotate")) or \
                                 (back and direction == "backward")
                        if danger and not dock_spin:
                            self._gap_lock_until = max(self._gap_lock_until, time.time() + 3.0)
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
        "thinking": ["THINK", "CONCENTRATE", "SCAN"],
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
        if not self.has.get("arm"):
            return False
        try:
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
        self.arms_up()
        time.sleep(0.3)
        self.arm_angle(90)
        return True

    # ----------------------------------------------------------------- drive
    def drive_distance(self, mm, speed=45):
        if not self.has.get("drive"):
            return False
        if not self.ensure_mobility():
            _log("drive blocked: docked")
            return False
        if not self._motion_allowed("forward" if mm >= 0 else "backward"):
            return False
        try:
            # SDK distance is unsigned; direction is the to_forward flag
            self._drive.go_distance(self._next_id(), abs(mm), speed, mm >= 0, True)
            self._watch_motion("forward" if mm >= 0 else "backward")
            return True
        except Exception as e:
            _log(f"drive_distance failed: {e}")
            return False

    def drive_rotate(self, degrees, speed=45):
        if not self.has.get("drive"):
            return False
        if getattr(self, "docked", False):
            # rotation ON the dock is probe-proven safe (she pivots in place);
            # only linear motion risks rolling off the plate
            _log("rotate allowed while docked (in-place pivot)")
        elif not self.ensure_mobility():
            _log("rotate blocked: docked")
            return False
        if not self._motion_allowed("rotate"):
            return False
        try:
            self._drive.go_rotate(self._next_id(), degrees, False, speed, True, True)
            self._watch_motion("rotate")
            return True
        except Exception as e:
            _log(f"drive_rotate failed: {e}")
            return False

    def drive_stop(self):
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

    _DANCES = {
        "fiesta": {"music": "/.doly/sounds/music/salsa.wav", "sfx": None,
                   "colors": ("Magenta", "Cyan", "Yellow", "Green"),
                   "moves": ("spin_big",)},
        "groove": {"music": "/.doly/sounds/sfx/drum roll (1).wav", "sfx": None,
                   "colors": ("Cyan", "Blue", "Purple"),
                   "moves": ("arms_wave", "wiggle")},
        "party": {"music": "/.doly/sounds/music/birthday.wav", "sfx": "collect",
                  "colors": ("Yellow", "Pink", "Orange", "Cyan"),
                  "moves": ("bounce", "spin_small", "arms_wave")},
    }

    def dance(self, variant=None):
        """Themed dance: music, lights, arms, spins. Preflight FIRST."""
        import random
        if not self._motion_allowed("rotate"):
            _log("dance blocked before show: preflight failed")
            return False
        name = variant if variant in self._DANCES else random.choice(list(self._DANCES))
        d = self._DANCES[name]
        docked = not self.ensure_mobility()
        _log(f"dance: {name}{' (dock mode: arms only)' if docked else ''}")
        if d["music"]:
            self.play_sfx(d["music"], defer=False)
        if d["sfx"]:
            sm = self.cfg.get("sounds", {}).get("sfx_map", {})
            self.play_sfx(sm.get(d["sfx"], ""), defer=False)
        for color in d["colors"]:
            self._led_flash(color)
            time.sleep(0.2)
        self.eyes("speaking")
        ok = True
        if docked:
            # dock party: everything but wheels
            for ang in (150, 30, 150, 30, 140, 40):
                self.arm_angle(ang, speed=75)
            self.mood_eyes("HEARTS")
            self._bump_mood(1)
            return True
        for move in d["moves"]:
            if not self._motion_allowed("rotate"):
                return False
            if move == "spin_big":
                ok &= self.drive_rotate(90, speed=40)
                ok &= self.arm_angle(150, speed=60)
                ok &= self.drive_rotate(-180, speed=40)
                ok &= self.arm_angle(20, speed=60)
                ok &= self.drive_rotate(90, speed=40)
            elif move == "spin_small":
                ok &= self.drive_rotate(60, speed=35)
                ok &= self.drive_rotate(-120, speed=35)
            elif move == "arms_wave":
                for ang in (150, 30, 150, 30):
                    ok &= self.arm_angle(ang, speed=70)
            elif move == "wiggle":
                ok &= self.drive_rotate(25, speed=45)
                ok &= self.drive_rotate(-50, speed=45)
                ok &= self.drive_rotate(25, speed=45)
            elif move == "bounce":
                for _ in range(2):
                    ok &= self.arm_angle(140, speed=80)
                    ok &= self.arm_angle(40, speed=80)
        ok &= self.arm_angle(20, speed=60)
        self._bump_mood(1)
        return ok

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
        """A little life while waiting: curious glance, sometimes a chirp."""
        import random
        try:
            self.mood_eyes(random.choice(self._CURIOS))
            if random.random() < 0.35:
                sfx_map = self.cfg.get("sounds", {}).get("sfx_map", {})
                pool = [sfx_map.get("pet", ""), sfx_map.get("click", ""),
                        self.cfg.get("wake", {}).get("chirp", "")]
                self.play_sfx(random.choice([p for p in pool if p]) or "", defer=False)
        except Exception as e:
            _log(f"idle_flourish: {e}")

    def wander_step(self):
        """Pet-like exploration: ONE safe move + a curious look.
        Short, preflighted, edge-gated, battery-aware."""
        import random
        try:
            pct = self.battery_pct()
            if pct is not None and pct < 20:
                return False
            if not self._motion_allowed("rotate"):
                return False
            # the probe can read "not docked" on the grippy charger plate
            # (wheels bite, yaw moves) — all-four void is the reliable signal
            on_dock = getattr(self, "docked", False) or len(self._edge_gaps()) >= 4
            if on_dock:
                # on the dock: eyes and arms only, never wheels
                if random.random() < 0.5:
                    self.idle_flourish()
                else:
                    self.arm_angle(120, speed=45)
                    time.sleep(0.3)
                    self.arm_angle(20, speed=45)
                    self.idle_flourish()
                return True
            move = random.choice(["look", "turn", "scoot", "turn", "scoot"])
            if move == "look":
                self.idle_flourish()
                return True
            if move == "turn":
                self.drive_rotate(random.choice([-90, -60, 60, 90]), speed=35)
            elif move == "scoot":
                self.drive_distance(random.choice([60, 90, 120]), speed=35)
            self.idle_flourish()
            return True
        except Exception as e:
            _log(f"wander_step: {e}")
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
        self.eyes("sleepy")
        self.arms_down()
        self.led_color("Black")

    # --------------------------------------------------------------- cleanup
    def dispose(self):
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
