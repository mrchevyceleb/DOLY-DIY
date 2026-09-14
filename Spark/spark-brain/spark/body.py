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

        imu.on_gesture(_on_gesture)
        self._imu = imu

    _TOF_REACTIONS = {
        "ObjectComing": ("CAUTIOUS", None, "back"),   # hand close → back away
        "ObjectGoing": ("HAPPY", None, "forward"),      # hand back → come on
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
            # EMERGENCY: kill motion, lock further motion briefly, react
            self._gap_lock_until = time.time() + 3.0
            _log(f"GAP DETECTED dir={direction} — motion locked")
            try:
                self.drive_stop()
            except Exception:
                pass
            self.eyes("thinking")  # closest to shocked; SCAN fallback
            self._led_flash("Red")

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
                    time.sleep(self._wav_duration(TTS_WAV) + 0.15)
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

    def pet_pulse(self):
        """Instant 'I felt that' reaction: sfx chirp + LED flash + happy eyes."""
        sfx = self.cfg.get("sounds", {}).get("pet_sfx")
        if sfx:
            self.play_sfx(sfx)
        self._led_flash("Cyan")
        self.eyes("listening")

    # ------------------------------------------------------------- edge lock
    def _motion_allowed(self):
        if not self.has.get("edge"):
            return True  # no sensors = can't gate (shouldn't happen)
        if time.time() < getattr(self, "_gap_lock_until", 0):
            _log("motion blocked: gap lock active")
            return False
        return True

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

    def play_sfx(self, path):
        if not self.has.get("sound"):
            return False
        try:
            self._snd.play(path, self._next_id())
            return True
        except Exception as e:
            _log(f"play_sfx failed: {e}")
            return False
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
        if not self._motion_allowed():
            return False
        try:
            # SDK distance is unsigned; direction is the to_forward flag
            self._drive.go_distance(self._next_id(), abs(mm), speed, mm >= 0, True)
            return True
        except Exception as e:
            _log(f"drive_distance failed: {e}")
            return False

    def drive_rotate(self, degrees, speed=45):
        if not self.has.get("drive"):
            return False
        if not self._motion_allowed():
            return False
        try:
            self._drive.go_rotate(self._next_id(), degrees, False, speed, True, True)
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

    def dance(self):
        """The full show: music, lights, arms, spins. Edge-gated."""
        if not self._motion_allowed():
            _log("dance blocked: too close to an edge")
            return False
        music = self.cfg.get("sounds", {}).get("dance_music")
        if music:
            self.play_sfx(music)
        # light show: quick color cycle
        for color in ("Magenta", "Cyan", "Yellow", "Green"):
            self._led_flash(color)
            time.sleep(0.25)
        self.eyes("speaking")
        self.arm_angle(150, speed=60)
        ok = True
        ok &= self.drive_rotate(120, speed=40)
        self.arm_angle(20, speed=60)
        ok &= self.drive_rotate(-240, speed=40)
        self.arm_angle(150, speed=60)
        ok &= self.drive_rotate(120, speed=40)
        self.arm_angle(20, speed=60)
        return ok

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
