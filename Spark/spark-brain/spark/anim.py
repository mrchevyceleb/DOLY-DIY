"""Stock Blockly playback through Body's safety gates, always on the main thread.

Per https://doly.ai/blockly/, start=1 waits for the PREVIOUS block and start=0
overlaps it. Distances are millimetres, rotations degrees, speeds percentages.
Cooperative jobs keep music, lights and motion together without SDK workers.
"""
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET

_SOUND_DIRS = {"DOLY": "sound", "MUSIC": "music", "SFX": "sfx", "ANIMAL": "animal", "WUW": "wuw"}
_HEX_TO_COLORCODE = {
    "#000000": "Black", "#ffffff": "White", "#ff0000": "Red", "#ff6600": "Orange",
    "#cc33cc": "Magenta", "#00ff00": "Lime", "#400000": "DarkRed", "#ffff00": "Yellow",
    "#00cccc": "Cyan", "#66ffff": "SkyBlue", "#666666": "Gray", "#6600cc": "Purple",
    "#0000ff": "Blue", "#ffcc00": "Gold", "#ff00ff": "Magenta", "#993399": "Purple",
    "#ff99ff": "Pink", "#9999ff": "SkyBlue", "#33ff33": "Green", "#333333": "Gray",
}


def _log(msg):
    print(f"[anim] {msg}", file=sys.stderr, flush=True)


def _fields(block):
    return {f.get("name"): (f.text or "") for f in block.findall("field")}


def _statement(block, name):
    st = block.find(f"statement[@name='{name}']")
    return st.findall("block") if st is not None else []


class _Completion:
    def __init__(self, ready=lambda: True):
        self.ready = ready
        self.finished = False
        self.cancelled = False

    def done(self):
        if not self.finished:
            self.finished = self.cancelled or self.ready()
        return self.finished


class AnimPlayer:
    def __init__(self, body, cfg):
        self.body = body
        self.dir = cfg.get("animations", {}).get("dir", "/.doly/config/animations")
        self._stop = threading.Event()
        self._playing = False
        self.petting = False
        self._sound_cache = {}
        self._jobs, self._pending, self._resources = [], [], {}

    def stop(self):
        self._stop.set()

    def playing(self):
        return self._playing

    def play(self, name, blocking=True):
        """False on missing/empty programs, execution failure or cancellation.

        Sensor callbacks use Body.queue_anim; native sounds must run on the
        main thread alongside recognition, never race Vosk on another thread.
        """
        if (not blocking or threading.current_thread() is not threading.main_thread()
                or self._playing or not re.fullmatch(r"[a-zA-Z0-9_]+", name)):
            _log(f"refused concurrent/background or invalid animation {name!r}")
            return False
        try:
            root = ET.parse(os.path.join(self.dir, f"{name}.xml")).getroot()
            for element in root.iter():
                element.tag = element.tag.rsplit("}", 1)[-1]
            blocks = root.findall("block")
            if not any(b.get("type") != "start_animation" for b in blocks):
                raise ValueError("no executable blocks")
            # Validate nested callbacks before any actuator dispatch.
            for block in root.iter("block"):
                kind = block.get("type")
                if kind not in ("start_animation", "led_animation_color") and not hasattr(self, f"_b_{kind}"):
                    raise ValueError(f"unsupported block {kind}")
        except Exception as e:
            _log(f"load {name}: {e}")
            return False
        self._stop.clear()
        self._jobs, self._pending, self._resources = [], [], {}
        self._playing = True
        # Continued stroking is allowed only for the motor-free stock pet
        # reactions. A modified pet file with any motor block still stops.
        self.petting = (name in {"petting1", "petting2", "petting3"}
                        and not any(b.get("type", "").startswith(("drive_", "arm_"))
                                    for b in root.iter("block")))
        success = False
        started = time.monotonic()
        try:
            self._spawn(self._sequence(blocks))
            while (self._jobs or self._pending) and not self._stop.is_set():
                if time.monotonic() - started > 180:
                    raise TimeoutError("animation exceeded three minutes")
                for job in list(self._jobs):
                    iterator, waiting, result = job
                    if result.cancelled:
                        self._jobs.remove(job)
                    elif waiting.done():
                        try:
                            job[1] = next(iterator)
                        except StopIteration:
                            result.finished = True
                            self._jobs.remove(job)
                self._pending = [p for p in self._pending if not p.done()]
                if self._jobs or self._pending:
                    self._stop.wait(0.01)
            success = not self._stop.is_set()
            _log(f"{name}: {'completed' if success else 'cancelled'} in {time.monotonic()-started:.2f}s")
        except Exception as e:
            _log(f"{name} failed: {e}")
        finally:
            if not success:
                self.body.stop_everything()
                if self.body.has.get("sound"):
                    try:
                        self.body._snd.abort()
                        self.body._speaking_until = time.time() + 0.25
                    except Exception as e:
                        _log(f"sound stop: {e}")
            self._jobs, self._pending, self._resources = [], [], {}
            self._playing = False
            self.petting = False
        return success

    def _spawn(self, iterator):
        result = _Completion(lambda: False)
        self._jobs.append([iterator, _Completion(), result])
        return result

    def _track(self, ready, resources=()):
        result = _Completion(ready)
        self._pending.append(result)
        for resource in resources:
            old = self._resources.get(resource)
            if old is not None and not old.done():
                old.cancelled = True
            self._resources[resource] = result
        return result

    def _delay(self, seconds):
        deadline = time.monotonic() + max(0, seconds)
        return self._track(lambda: time.monotonic() >= deadline)

    def _sequence(self, blocks):
        previous = _Completion()
        for block in blocks:
            if self._stop.is_set():
                return
            if block.get("type") == "start_animation":
                continue
            fields = _fields(block)
            if fields.get("start") == "1":
                yield previous
                if self._stop.is_set():
                    return
            previous = getattr(self, f"_b_{block.get('type')}")(fields, block) or _Completion()
        yield previous

    def _b_delay_ms(self, f, block):
        return self._delay(float(f.get("delay_ms", "0")) / 1000)

    def _b_repeat(self, f, block):
        def repeat():
            for _ in range(max(0, min(100, int(f.get("times", "1"))))):
                yield self._spawn(self._sequence(_statement(block, "repeat_statement")))
        return self._spawn(repeat())

    def _b_eye_animations(self, f, block):
        if not self.body.has.get("eye"):
            return
        name = f.get("animation", "").upper().replace(" ", "_")
        if self.body.mood_eyes("NERVOUS" if name == "SHAKY" else name) is False:
            raise RuntimeError(f"cannot show expression {name}")
        return self._state_completion(lambda: self.body._eye.is_animating(), 15, ("eye",))

    def _b_eye_background(self, f, block):
        # Stock love uses a HEARTS background. Match its expression without
        # overwriting the user's persistent eye background.
        return self._b_eye_animations({"animation": f.get("style", "HEARTS")}, block)

    def _b_speak(self, f, block):
        if self.body.speak(f.get("say", ""), wait=False) is False:
            raise RuntimeError("cannot speak animation text")
        return self._track(lambda: not self.body.speaking_recently(), ("sound",))

    def _b_sound(self, f, block):
        path = self._sound_path(f.get("type_id", "DOLY"), f.get("name", ""))
        if not path or not self.body.play_sfx(path, defer=False):
            raise RuntimeError(f"cannot play sound {f.get('name')}")
        duration = self.body._wav_duration(path)
        self.body._speaking_until = time.time() + duration + 0.25
        deadline = time.monotonic() + duration
        sound = self._track(lambda: time.monotonic() >= deadline, ("sound",))
        complete = _statement(block, "complete_statement")
        if complete:
            def after_sound():
                yield sound
                if not sound.cancelled:
                    yield from self._sequence(complete)
            self._spawn(after_sound())
        return sound

    def _b_led(self, f, block):
        sides = (1, 2) if f.get("side", "0") == "0" else (int(f["side"]),)
        for side in sides:
            old = self._resources.get(f"led{side}")
            if old:
                old.cancelled = True
        self._led_solid(f.get("color_main", "#000000"), int(f.get("side", "0")))

    def _b_led_animation(self, f, block):
        def run_side(side, steps):
            for step in steps:
                st = _fields(step)
                ms = max(0, int(st.get("led_time", "500")))
                self._led_solid(st.get("color_main", "#000000"), side, st.get("color_fade"), ms)
                yield self._delay(ms / 1000)
        jobs = []
        for name, side in (("led_left", 1), ("led_right", 2)):
            old = self._resources.get(f"led{side}")
            if old:
                old.cancelled = True
            job = self._spawn(run_side(side, _statement(block, name)))
            self._resources[f"led{side}"] = job
            jobs.append(job)
        return _Completion(lambda: all(j.done() for j in jobs))

    def _b_arm_set_angle(self, f, block):
        return self._arm_to(int(f.get("side", "0")), round(float(f.get("angle", "90"))),
                            self._arm_speed(int(f.get("speed", "40"))))

    def _b_drive_distance(self, f, block):
        mm = float(f.get("distance", "0")) * (-1 if f.get("direction") == "0" else 1)
        if self.body.drive_distance(mm, speed=self._drive_speed(int(f.get("speed", "20")))):
            return self._drive_completion()

    def _b_drive_rotate_left(self, f, block):
        return self._rotate(f, -1)

    def _b_drive_rotate_right(self, f, block):
        return self._rotate(f, 1)

    def _rotate(self, f, sign):
        if self.body.drive_rotate(sign * float(f.get("driveRotate", "0")),
                                  speed=self._drive_speed(int(f.get("speed", "20"))),
                                  from_center=f.get("isCenter", "TRUE").upper() == "TRUE"):
            return self._drive_completion()

    @staticmethod
    def _arm_speed(speed):
        return max(1, min(80, speed))

    @staticmethod
    def _drive_speed(speed):
        return max(1, min(40, speed))

    def _state_completion(self, running, timeout, resources):
        deadline = time.monotonic() + timeout
        def ready():
            if not running():
                return True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{resources} did not finish")
            return False
        return self._track(ready, resources)

    def _drive_completion(self):
        d = self.body._drive
        def running():
            state = d.get_state()
            if state == d.DriveState.Error:
                raise RuntimeError("drive controller error")
            return state == d.DriveState.Running
        return self._state_completion(running, 15, ("drive",))

    def _arm_to(self, side_xml, angle, speed, wait=False):
        b = self.body
        if b.actuators_held() or not b.has.get("arm"):
            return
        side = {0: b._arm.ArmSide.Both, 1: b._arm.ArmSide.Left, 2: b._arm.ArmSide.Right}[side_xml]
        with b._power_lock:
            if b.actuators_held() or self._stop.is_set():
                return
            rc = b._arm.set_angle(b._next_id(), side, speed=int(speed), angle=int(round(angle)), with_brake=False)
        if rc < 0:
            raise RuntimeError(f"arm rc={rc} (angle {angle}, speed {speed})")
        def running(arm_side):
            state = b._arm.get_state(arm_side)
            if state == b._arm.ArmState.Error:
                raise RuntimeError("arm controller error")
            return state == b._arm.ArmState.Running
        # Replacing one side must not discard completion of the other arm.
        parts = []
        for side_num, arm_side in ((1, b._arm.ArmSide.Left), (2, b._arm.ArmSide.Right)):
            if side_xml in (0, side_num):
                parts.append(self._state_completion(lambda s=arm_side: running(s), 10, (f"arm{side_num}",)))
        return _Completion(lambda: all(p.done() for p in parts))

    def _led_solid(self, hex_color, side_xml, fade=None, ms=0):
        b = self.body
        if not b.has.get("led"):
            return
        name = _HEX_TO_COLORCODE.get((hex_color or "").lower())
        fade_name = _HEX_TO_COLORCODE.get((fade or "").lower()) if fade else None
        if name is None:
            return
        try:
            sides = {0: [b._led.LedSide.Both],
                     1: [b._led.LedSide.Left], 2: [b._led.LedSide.Right]}[side_xml]
            activity = b._led.LedActivity()
            activity.mainColor = b._color.from_code(getattr(b._ledcc, name, b._ledcc.Black))
            if fade_name:
                activity.fadeColor = b._color.from_code(getattr(b._ledcc, fade_name))
                activity.fade_time = ms
            for s in sides:
                b._led.process_activity(b._next_id(), s, activity)
        except Exception as e:
            raise RuntimeError(f"led: {e}") from e

    def _sound_path(self, type_id, name):
        key = (type_id, name)
        if key in self._sound_cache:
            return self._sound_cache[key]
        sub = _SOUND_DIRS.get((type_id or "").upper(), "sound")
        base = f"/.doly/sounds/{sub}"
        want = (name or "").strip().lower() + ".wav"
        found = None
        if want != ".wav":
            candidate = os.path.join(base, want)
            if os.path.exists(candidate):
                found = candidate
            else:  # case-insensitive fallback scan
                try:
                    for fn in os.listdir(base):
                        if fn.lower() == want:
                            found = os.path.join(base, fn)
                            break
                except Exception:
                    pass
        if not found:
            _log(f"sound not found: {type_id}/{name}")
        self._sound_cache[key] = found
        return found
