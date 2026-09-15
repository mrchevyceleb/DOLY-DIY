"""Stock Blockly animation player.

Doly's stock behaviors live in /.doly/config/animations/*.xml as Blockly
programs. This interpreter executes them against Body, so Spark inherits the
whole stock library (salsa, fist bump, high five, petting, sleep, wakeup...)
with Spark's safety layer still wrapped around every drive command.

Block semantics (observed from the stock XML files):
- Blocks run in document order.
- field start=1 -> wait for THIS block to finish before continuing
  (sound plays out, arm reaches angle, drive completes, delay elapses).
- start=0 -> fire and continue immediately.
- repeat {times, repeat_statement}: sequential loop.
- sound {type_id, name}: DOLY -> /.doly/sounds/sound/<name>.wav,
  MUSIC -> /.doly/sounds/music/<name>.wav, SFX -> /.doly/sounds/sfx/<name>.wav
  (name lowercased). Optional complete_statement runs when the sound ends.
- led {color_main, side}: solid color; side 0=Both 1=Left 2=Right.
- led_animation {led_left/led_right}: per-side sequences of
  led_animation_color {color_main, led_time, color_fade} steps.
- arm_set_angle {angle, side, speed, brake}.
- drive_distance {distance (mm), direction (1=fwd, 0=back), speed, accel, brake}.
- drive_rotate_left/right {driveRotate (x10 = degrees), isCenter, speed, accel, brake}.
- delay_ms {delay_ms}.
- eye_animations {category, animation}: the animation name uppercased with
  spaces -> underscores maps onto doly_eye.expressions members.
"""
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET

_SOUND_DIRS = {"DOLY": "sound", "MUSIC": "music", "SFX": "sfx", "ANIMAL": "animal", "WUW": "wuw"}

# Blockly XML colors -> doly ColorCode names (the ~20 values the stock
# animations actually use, per a full scan of the animation directory).
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


class AnimPlayer:
    def __init__(self, body, cfg):
        self.body = body
        self.dir = cfg.get("animations", {}).get("dir", "/.doly/config/animations")
        self._stop = threading.Event()
        self._thread = None
        self._sound_cache = {}

    # ------------------------------------------------------------- lifecycle
    def stop(self):
        self._stop.set()

    def playing(self):
        return bool(self._thread and self._thread.is_alive())

    def play(self, name, blocking=True):
        """Play an animation by file stem ('salsa', 'fist_bump', ...)."""
        path = os.path.join(self.dir, f"{name}.xml")
        if not os.path.exists(path):
            _log(f"missing animation {name}")
            return False
        try:
            blocks = ET.parse(path).getroot().findall("block")
        except Exception as e:
            _log(f"parse {name}: {e}")
            return False
        self.stop()
        if self._thread:
            self._thread.join(timeout=2)
        self._stop.clear()

        def _run():
            try:
                self._run_blocks(blocks)
            except Exception as e:
                _log(f"{name} failed: {e}")

        if blocking:
            _run()
            return True
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    # -------------------------------------------------------------- executor
    def _run_blocks(self, blocks):
        for block in blocks:
            if self._stop.is_set():
                return
            self._run_block(block)

    def _run_block(self, block):
        btype = block.get("type")
        f = _fields(block)
        blocking = f.get("start") == "1"
        handler = getattr(self, f"_b_{btype}", None)
        if handler is None and btype not in ("start_animation",):
            _log(f"unknown block {btype}")
        if handler:
            handler(f, blocking, block)

    # --------------------------------------------------------------- blocks
    def _b_delay_ms(self, f, blocking, block):
        try:
            time.sleep(max(0, int(f.get("delay_ms", "0"))) / 1000.0)
        except Exception:
            pass

    def _b_repeat(self, f, blocking, block):
        times = max(1, int(f.get("times", "1")))
        inner = _statement(block, "repeat_statement")
        for _ in range(times):
            if self._stop.is_set():
                return
            self._run_blocks(inner)

    def _b_eye_animations(self, f, blocking, block):
        name = (f.get("animation") or "").upper().replace(" ", "_")
        if name:
            self.body.mood_eyes(name)

    def _b_sound(self, f, blocking, block):
        path = self._sound_path(f.get("type_id", "DOLY"), f.get("name", ""))
        complete = _statement(block, "complete_statement")
        if not path:
            if complete:
                self._run_blocks(complete)
            return
        self.body.play_sfx(path, defer=False)
        if blocking or complete:
            time.sleep(self.body._wav_duration(path) + 0.1)
        if complete and not self._stop.is_set():
            self._run_blocks(complete)

    def _b_led(self, f, blocking, block):
        self._led_solid(f.get("color_main", "#000000"), int(f.get("side", "0")))

    def _b_led_animation(self, f, blocking, block):
        seqs = []
        for stmt_name, side in (("led_left", 1), ("led_right", 2)):
            steps = [_fields(b) for b in _statement(block, stmt_name)
                     if b.get("type") == "led_animation_color"]
            if steps:
                seqs.append((side, steps))

        def _run_side(side, steps):
            for st in steps:
                if self._stop.is_set():
                    return
                self._led_solid(st.get("color_main", "#000000"), side,
                                fade=st.get("color_fade"),
                                ms=int(st.get("led_time", "500")))
                time.sleep(int(st.get("led_time", "500")) / 1000.0)

        threads = [threading.Thread(target=_run_side, args=(s, st), daemon=True)
                   for s, st in seqs]
        for t in threads:
            t.start()
        if blocking:
            for t in threads:
                t.join()

    def _b_arm_set_angle(self, f, blocking, block):
        side_xml = int(f.get("side", "0"))
        angle = float(f.get("angle", "90"))
        speed = self._arm_speed(int(f.get("speed", "10")))
        self._arm_to(side_xml, angle, speed, wait=blocking)

    def _b_drive_distance(self, f, blocking, block):
        mm = float(f.get("distance", "0"))
        if f.get("direction", "1") == "0":
            mm = -mm
        speed = self._drive_speed(int(f.get("speed", "20")))
        self.body.drive_distance(mm, speed=speed)
        # ALWAYS wait for drive completion — an animation that returns while
        # the wheels still turn drops its safety flags (e.g. leaving-home)
        # and lets the watchdog kill the ritual mid-lip
        self._wait_drive()

    def _b_drive_rotate_left(self, f, blocking, block):
        self._rotate(f, sign=-1, blocking=blocking)

    def _b_drive_rotate_right(self, f, blocking, block):
        self._rotate(f, sign=1, blocking=blocking)

    # --------------------------------------------------------------- helpers
    def _rotate(self, f, sign, blocking):
        deg = sign * float(f.get("driveRotate", "0")) * 10.0  # units: x10 degrees
        speed = self._drive_speed(int(f.get("speed", "5")))
        self.body.drive_rotate(deg, speed=speed)
        self._wait_drive()  # always wait (see _b_drive_distance)

    @staticmethod
    def _arm_speed(xml_speed):
        return max(5, min(80, xml_speed * 4))  # stock 1..20 -> sdk 5..80

    @staticmethod
    def _drive_speed(xml_speed):
        return max(15, min(60, xml_speed * 2 + 10))  # stock 1..40 -> sdk 15..60

    def _wait_drive(self, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop.is_set():
            try:
                if self.body._drive.get_state() != self.body._drive.DriveState.Running:
                    return
            except Exception:
                return
            time.sleep(0.05)

    def _arm_to(self, side_xml, angle, speed, wait):
        b = self.body
        if not b.has.get("arm"):
            return
        side = {0: b._arm.ArmSide.Both, 1: b._arm.ArmSide.Left,
                2: b._arm.ArmSide.Right}[side_xml]
        try:
            rc = b._arm.set_angle(b._next_id(), side, speed=speed, angle=angle,
                                  with_brake=False)
            if rc < 0:
                _log(f"arm rc={rc} (angle {angle}, speed {speed})")
                return
            if wait:
                deadline = time.time() + 5
                while time.time() < deadline and not self._stop.is_set():
                    if b._arm.get_state(side) == b._arm.ArmState.Completed:
                        break
                    time.sleep(0.05)
        except Exception as e:
            _log(f"arm: {e}")

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
            _log(f"led: {e}")

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
