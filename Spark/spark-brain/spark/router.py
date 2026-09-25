"""Router — stock commands first, LLM for everything else.

Phase 1 policy:
- A stock command match executes locally (instant, works offline).
- Anything else goes to the brain on Moria.
- If the brain is offline, free speech degrades gracefully.
"""
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.request

from . import commands as cmds
from . import search as websearch
from .govee import GoveeLights
from .sched import AlarmClock, fmt_clock, next_occurrence


def _log(msg):
    print(f"[router] {msg}", file=sys.stderr)


# "Imagine..." -> she acts it out: themed routine + the brain narrating theatrically
_IMAGINE_RE = re.compile(r"\b(imagine|pretend|act like|act out)\b", re.IGNORECASE)
_IMAGINE_THEMES = {
    "beach": "fiesta", "party": "party", "exercise": "workout",
    "workout": "workout", "birthday": "party", "dance": "fiesta",
    "robot": "groove", "firefighter": "fireman", "fireman": "fireman",
    "police": "policeman", "meditat": "meditate", "rock": "rock",
}

# Explicit web-search intent — always goes to the search tool, not the stock table.
_SEARCH_INTENT = re.compile(
    r"\b(?:search (?:the web |the internet )?(?:for |about )?|look up|google|"
    r"what'?s the latest (?:on |with )?|any news (?:on |about )?|news about)\b",
    re.IGNORECASE,
)


# Govee room lights — "my/room/the lights", never her own LEDs/eyes
_GOVEE_RE = re.compile(r"\b(?:govee|lights?)\b", re.IGNORECASE)
_HERS_LIGHTS_RE = re.compile(r"\byour\s+(?:lights?|leds?)\b", re.IGNORECASE)
_GOVEE_OFF_RE = re.compile(r"\b(?:turn\s+off|switch\s+off|shut\s+off|lights?\s+off|kill|blackout)\b", re.I)
_GOVEE_ON_RE = re.compile(r"\b(?:turn\s+on|switch\s+on|lights?\s+on|put\s+on|fire\s+up)\b", re.I)

# alarms / timers / reminders — absolute and relative scheduling
_ALARM_WORD_RE = re.compile(r"\balarms?\b|\bwake me\b", re.I)
_ALARM_CANCEL_RE = re.compile(r"\b(?:cancel|clear|stop|kill|delete|forget)\b[^.]*\balarms?", re.I)
_ALARM_QUERY_RE = re.compile(r"\b(?:what|which|when|how many)\b[^.]*\balarms?|\balarms?\b[^.]*\b(?:status|set|do i have)\b", re.I)
_TIMER_CANCEL_RE = re.compile(r"\b(?:cancel|clear|stop|kill|delete|forget)\b[^.]*\btimer|\btimer\b[^.]*\b(?:off|cancel)\b", re.I)
_TIMER_QUERY_RE = re.compile(r"\b(?:how much|how long|time left|status)\b[^.]*\btimer|\btimer\b[^.]*\bleft\b", re.I)
_REMINDER_RE = re.compile(r"\bremind me\b", re.I)
_REMINDER_PARSE_RE = re.compile(r"remind me\s+(?:to\s+|about\s+|that\s+)?(.+?)\s+(?:in|after)\s+(.+)$", re.I)

EDGE_REFUSAL = ("I can't drive here — I'm either on my dock or too close to an edge. "
                "Put me somewhere with room and ask again!")
DOCK_REFUSAL = "I couldn't leave the charger safely, so I'm staying parked."

# Voice switching by name — aliases include common ASR mishearings
# ("switch to weekly" really is how "wheatley" comes back from the mic).
# alias -> (display, server_voice, local model|None, needs modern piper,
#           pitch semitones, robot mix)
# server_voice names resolve on Moria's piper server (~0.4s synth); None
# means the voice only exists locally on the Pi. The three curated picks:
# Robot = hfc +2st + sheen (Matt's default), hfc = +2st plain,
# lessac = +2st on the clearest voice in the catalog.
_MODULE_PY = "/opt/piper-ng/bin/python"
_PIPER_DIR = "/.doly/data/piper"
_VOICES = {
    "robot":    ("Robot",    "hfc",      f"{_PIPER_DIR}/en_US-hfc_female-medium.onnx", False, 2, 0.25),
    "default":  ("Robot",    "hfc",      f"{_PIPER_DIR}/en_US-hfc_female-medium.onnx", False, 2, 0.25),
    "spark":    ("Robot",    "hfc",      f"{_PIPER_DIR}/en_US-hfc_female-medium.onnx", False, 2, 0.25),
    "hfc":      ("HFC",      "hfc",      f"{_PIPER_DIR}/en_US-hfc_female-medium.onnx", False, 2, 0.0),
    "lessac":   ("Lessac",   "lessac",   f"{_PIPER_DIR}/en_US-lessac-high.onnx",       False, 2, 0.0),
    "kathleen": ("Kathleen", "kathleen", f"{_PIPER_DIR}/en_US-kathleen-low.onnx",      False, 3, 0.0),
    "cori":     ("Cori",     "cori",     f"{_PIPER_DIR}/en_GB-cori-high.onnx",         False, 2, 0.0),
    "wheatley": ("Wheatley", None,       f"{_PIPER_DIR}/wheatley-en.onnx",             True,  0, 0.0),
    "weekly":   ("Wheatley", None,       f"{_PIPER_DIR}/wheatley-en.onnx",             True,  0, 0.0),
    "glados":   ("GLaDOS",   None,       f"{_PIPER_DIR}/glados.onnx",                  False, 0, 0.0),
    "gladys":   ("GLaDOS",   None,       f"{_PIPER_DIR}/glados.onnx",                  False, 0, 0.0),
    "amy":      ("Amy",      None,       f"{_PIPER_DIR}/en_US-amy-medium.onnx",        False, 0, 0.0),
    "stock":    ("stock",    None,       None,                                         False, 0, 0.0),
    "original": ("stock",    None,       None,                                         False, 0, 0.0),
}
_CURATED = ("Robot", "HFC", "Lessac")
_SWITCH_VERB_RE = re.compile(r"\b(switch|change|swap|use|try)\b", re.IGNORECASE)
_VOICE_LIST_RE = re.compile(r"\b(what|which|list)\b", re.IGNORECASE)


def _voice_intent(low):
    """Voice tuple when the text asks to switch voices, else None."""
    if not (_SWITCH_VERB_RE.search(low) or "voice" in low):
        return None
    for alias, val in _VOICES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return val
    return None


class Router:
    def __init__(self, cfg, body, brain, memory):
        self.cfg = cfg
        self.body = body
        self.brain = brain
        self.memory = memory
        try:
            self.govee = GoveeLights(cfg)
        except Exception as e:
            _log(f"govee unavailable: {e}")
            self.govee = None
        try:
            self.alarms = AlarmClock(cfg, self._scheduled_fire)
        except Exception as e:
            _log(f"alarm clock unavailable: {e}")
            self.alarms = None
        self.llm_reply = None  # set by Spark: streamed brain reply w/ context
        self.last_motion_result = None
        self._last_motion_at = 0
        self._timers = []
        self._pending = None    # {"kind": "timer"|"alarm", "at": ts} awaiting the spoken answer

    # ------------------------------------------------------------------ main
    def handle(self, raw_text):
        """Returns True if handled as a local command (incl. web search)."""
        text = cmds.normalize(raw_text)
        if not text:
            self.body.speak("Yes?")
            return True

        # emergency stop outranks EVERY other route (search, colors, table)
        tokens = text.split()
        if tokens and (tokens[0] == "stop" or tokens[-1] == "stop"):
            self.body.stop_everything()
            return True

        # she asked "how long?" / "for what time?" — this utterance is the answer
        pending = getattr(self, "_pending", None)
        if pending and time.time() - pending.get("at", 0) < 45:
            if pending["kind"] == "timer":
                secs = cmds.parse_timer(text)
                if secs:
                    self._pending = None
                    alarms0 = getattr(self, "alarms", None)
                    if alarms0 is not None:
                        alarms0.add_timer(secs)
                    else:
                        self._start_timer(secs)
                    self.body.speak(self._describe_timer(secs) + ". I'm on it.")
                    return True
            elif pending["kind"] == "alarm":
                parsed = cmds.parse_clock_time(text)
                if parsed:
                    self._pending = None
                    return self._set_alarm_from(parsed)
            self._pending = None  # user moved on; route normally

        # imagine prompts: she physically acts it out while narrating
        if _IMAGINE_RE.search(raw_text) and not cmds._NEGATION_RE.search(text):
            return self._imagine(raw_text)

        # explicit web search → tool + brain-mediated answer
        if _SEARCH_INTENT.search(raw_text):
            self._web_search(raw_text)
            return True

        # alarms, timers, reminders — local scheduling before anything fuzzy
        low = text.lower()
        alarms = getattr(self, "alarms", None)
        if alarms is not None:
            if _REMINDER_RE.search(low):
                return self._set_reminder(text)
            if _TIMER_CANCEL_RE.search(low):
                n = alarms.cancel("timer")
                self.body.speak("Timer cancelled." if n else "You don't have a timer running.")
                return True
            if _TIMER_QUERY_RE.search(low):
                left = alarms.remaining("timer")
                self.body.speak(f"{alarms._human(left)} left on your timer." if left is not None
                                else "You don't have a timer running.")
                return True
            if _ALARM_WORD_RE.search(low):
                return self._alarms(text)

        # Govee room lights: instant local control, checked before her own LEDs
        low = text.lower()
        if (self.govee and self.govee.enabled and _GOVEE_RE.search(low)
                and not _HERS_LIGHTS_RE.search(low) and "eye" not in low):
            reply = self._govee_lights(text)
            if reply is not None:
                self.body.speak(reply)
                return True

        # color commands (need param extraction before fuzzy match)
        if ("eye" in low or "eyes" in low) and ("color" in low or "colour" in low):
            color = cmds.extract_color(text)
            if color and self.body.eye_color(color):
                self.body.speak(f"Ooh, {color.replace('_', ' ').lower()}. Nice choice.")
                return True
        if ("led" in low or "light" in low) and ("color" in low or "colour" in low):
            color = cmds.extract_color(text)
            if color and self.body.led_color(color):
                self.body.speak(f"LEDs going {color.lower()}.")
                return True

        # voice management: "switch to robot", "what voices do you have"
        voice = _voice_intent(low)
        if voice is not None:
            name, server_voice, model, needs_module, pitch, robot_mix = voice
            return self._voice_switch(name, server_voice, model, needs_module, pitch, robot_mix)
        if "voice" in low and _VOICE_LIST_RE.search(low):
            tts_cfg = self.cfg.get("tts", {})
            current = tts_cfg.get("voice_name") or tts_cfg.get("piper_model")
            current = os.path.basename(str(current)).split(".")[0] if current else "stock"
            b_names = ", ".join(sorted({v[0] for v in _VOICES.values() if v[0] != "stock"}))
            self.body.speak(f"I'm using my {current} voice. The tuned robot picks are "
                            f"{', '.join(_CURATED)}. I can also be: {b_names}, or stock. "
                            "Just say switch to, and a name.")
            return True

        cmd, score = cmds.match_command(raw_text)
        if cmd:
            _log(f"command={cmd['action']} score={score:.2f} text='{text}'")
            return self._execute(cmd["action"], text)

        if self._motion_followup(text):
            return True
        return False

    def _motion_followup(self, text):
        """Explain a failed approach from controller state, not chat fiction."""
        dock_correction = bool(cmds._DOCK_CORRECTION_RE.match(text))
        recent = (self.last_motion_result is not None
                  and time.monotonic()-self._last_motion_at < 90)
        correction = re.search(
            r"\b(?:i'm|i am) (?:right )?here\b|\bonly (?:one|1) person\b|"
            r"\b(?:you didn't (?:move|spin)|you did not (?:move|spin)|you're not moving|that didn't work)\b", text)
        if not dock_correction and not (recent and correction):
            return False
        charging = self.body.refresh_power()
        if self.body.docked:
            reply = "I'm still parked. My safe departure check hasn't cleared."
        elif charging is None:
            reply = "My power reading is uncertain, so I'm staying still."
        elif dock_correction:
            reply = "You're right, I'm off the charger. Say come here to try again."
        else:
            reason = self.last_motion_result["result"]
            reply = {
                "ambiguous": "My camera couldn't pick a target. Say come here to retry.",
                "lost": "My camera lost the target. Say come here to retry.",
                "sensor": "My obstacle readings weren't clear, so I stopped.",
                "not_found": "My camera didn't find you. Say come here to try again.",
                "edge": "The floor sensors stopped me because they detected a gap.",
                "stopped_edge": "The floor sensors stopped me because they detected a gap.",
                "cancelled": "I stopped when the stop control was triggered.",
                "not_started": "The motors didn't start that turn. I've stopped.",
                "error": "The motor controller reported an error, so I stopped.",
                "timeout": "The turn didn't finish in time, so I stopped.",
                "blocked": "The safety check blocked that movement.",
                "ok": "The motor controller reported completion, but I can't confirm the result you saw.",
            }.get(reason, "I'm stopped. Please ask again if you want me to retry.")
        _log(f"movement follow-up: {self.last_motion_result}")
        self.body.speak(reply)
        return True

    # ---------------------------------------------------------------- govee
    def _govee_lights(self, text):
        """Parse a room-lights request; None lets other handlers try."""
        low = text.lower()
        g = self.govee
        # on / off
        if _GOVEE_OFF_RE.search(low):
            return g.turn(False)
        if _GOVEE_ON_RE.search(low):
            return g.turn(True)
        # brightness: "dim (to N%)", "set to 40", "half", word numbers
        m = re.search(r"(?:dim|brighten|brightness|set|turn)[^0-9]{0,20}"
                      r"(\d+|" + "|".join(cmds._NUM_WORDS) + r")\s*(?:%|percent)?\b", low)
        if m and re.search(r"\b(?:dim|brighten|brightness|percent)\b", low):
            pct = m.group(1)
            pct = cmds._NUM_WORDS.get(pct, pct)
            try:
                return g.brightness(round(float(pct)))
            except (TypeError, ValueError):
                pass
        if re.search(r"\bdim\b", low):
            return g.brightness(30)
        if re.search(r"\bbrighten\b|\bfull\b", low):
            return g.brightness(100 if re.search(r"\bfull\b", low) else 75)
        # white temperatures
        if re.search(r"\bwarm\s*white\b|\bwarmer\b", low):
            return g.color_temp(3200)
        if re.search(r"\bcool\s*white\b|\bdaylight\b|\bcooler\b|\bwhiter\b", low):
            return g.color_temp(6500)
        # named color: "set my lights to blue", "make the lights purple"
        color = cmds.extract_color(text)
        if color:
            return g.color(color)
        # status / discovery check
        if re.search(r"\b(?:status|see|find|which|how many)\b", low):
            return g.status()
        if re.search(r"\blights?\s*$", low.strip()):
            return g.status()
        return None

    # -------------------------------------------------------------- executor
    def _execute(self, action, text):
        b = self.body

        # Only a matched, affirmative movement command authorizes departure.
        # Petting, chatter, battery questions and idle reactions stay parked.
        moving = (action in {"forward", "back", "left", "right", "spin", "come_here", "dance"}
                  or action.startswith("dance_"))
        if moving:
            b.refresh_power()
            if b.docked and not b._undock():
                result = b.last_departure_result
                self.last_motion_result = {"command": action, "result": result}
                self._last_motion_at = time.monotonic()
                b.speak("Stopped." if result == "cancelled" else
                        "I couldn't leave the charger safely, so I've stopped.")
                return True

        if action == "time":
            now = datetime.datetime.now().strftime("%I:%M")
            hour, minute = now.lstrip("0").split(":")
            minute = f"oh {minute}" if minute.startswith("0") else minute
            b.speak(f"It's {hour} {minute}.")
            return True

        if action == "date":
            today = datetime.datetime.now().strftime("%A, %B %d")
            b.speak(f"Today is {today}.")
            return True

        if action == "timer":
            secs = cmds.parse_timer(text)
            if not secs:
                self._pending = {"kind": "timer", "at": time.time()}
                b.speak("How long should I set it for?")
                return True
            if self.alarms is not None:
                self.alarms.add_timer(secs)
            else:
                self._start_timer(secs)
            b.speak(self._describe_timer(secs) + ". I'm on it.")
            return True

        if action == "weather":
            report = self._weather()
            b.speak(report or "I can't reach the weather service right now.")
            return True

        if action == "photo":
            b.eyes("thinking")
            path = b.take_photo()
            b.speak("Got it!" if path else "Hmm, my camera didn't cooperate.")
            return True

        if action == "fist_bump":
            b.speak("Give me a fist bump!")
            if not b.fist_bump():
                b.speak("I can’t raise my arms safely here. Put me on a clear surface first.")
            return True

        if action == "high_five":
            b.speak("Up top!")
            if not b.high_five():
                b.speak("I can’t raise my arms safely here. Put me on a clear surface first.")
            return True

        if action == "go_home":
            return self.go_home_action()

        if action == "dance" or action.startswith("dance_"):
            b.speak("Watch this.")
            variant = action[6:] if action.startswith("dance_") else None
            if not b.dance(variant):
                b.speak("That routine didn't finish. Let's try again later.")
            return True

        if action == "come_here":
            # No promise of movement until the camera and guards authorize it.
            result = b.come_here()
            self.last_motion_result = {"command": "come_here", "result": result}
            self._last_motion_at = time.monotonic()
            replies = {
                "near": "I'll stop here. Hello!",
                "limit": "I've stopped here. Call me again if you want me closer.",
                "docked": DOCK_REFUSAL,
                "power": "I need to charge before I can come over.",
                "not_found": "I couldn't see you. Step into view and call me again.",
                "lost": "I lost sight of you, so I stopped.",
                "ambiguous": "I couldn't pick a single person to approach.",
                "obstacle": "Something's in my way. I've stopped.",
                "sensor": "I can't get a clear reading from my obstacle sensors, so I'm staying here.",
                "edge": "There's an edge here. I've stopped.",
                "stopped_edge": "There's an edge here. I've stopped.",
                "cancelled": "Stopped.",
                "busy": "I'm already looking for you.",
                "unavailable": "My camera search isn't available right now.",
            }
            b.speak(replies.get(result, "I couldn't move safely, so I've stopped."))
            return True

        if action in ("spin", "left", "right"):
            result = b.turn_guarded({"spin": 360, "left": -90, "right": 90}[action])
            self.last_motion_result = {"command": action, "result": result}
            self._last_motion_at = time.monotonic()
            if result == "cancelled":
                b.speak("Stopped.")
            elif result != "ok":
                b.speak(DOCK_REFUSAL if b.docked else
                        "I couldn't finish that turn safely, so I stopped.")
            return True

        def _refuse():
            b.speak(DOCK_REFUSAL if getattr(b, "docked", False) else EDGE_REFUSAL)

        def _guarded(mm, speed):
            result = b.drive_guarded(mm, speed=speed)
            if result == "stopped_edge":
                if mm > 0:
                    b.speak("Whoa, that's the edge — backing up.")
                    b._escape_edge()
                else:
                    b.speak("There's an edge behind me — not going further.")
            elif not result:
                _refuse()
            return True

        if action == "forward":
            return _guarded(150, 30)
        if action == "back":
            return _guarded(-150, 30)
        if action == "stop":
            b.stop_everything()
            return True

        if action == "sleep":
            b.speak("Goodnight. Say Spark or tap me to wake me.")
            b.sleep_pose()
            return True

        if action == "wake":
            b.wake_up()
            b.speak("I'm awake!")
            return True

        if action == "battery":
            charging = b.refresh_power()
            pct = b.battery_pct()
            if pct is None:
                b.speak("I can't read my battery right now.")
            else:
                state = (" I'm charging." if charging is True else
                         " I'm parked, but not charging. Please reseat me." if b.docked and charging is False else
                         " My charging reading is uncertain." if charging is None else "")
                b.speak(f"I'm at about {pct} percent." + state)
            return True

        _log(f"unhandled action {action}")
        return False

    # ----------------------------------------------------------------- homing
    def go_home_action(self):
        b = self.body
        result = b.go_home()
        if result == "already":
            charging = b.refresh_power()
            b.speak("I'm already charging." if charging is True else
                    "I'm parked, but charging isn't confirmed. Please check my dock contact.")
        elif result == "arrived":
            b.speak("Home sweet home. Charging up!")
        elif result == "unknown":
            b.speak("I need help getting onto my charger. Please place me on my dock.")
        elif result == "busy":
            b.speak("I'm already heading home!")
        elif result == "cancelled":
            b.speak("Stopped.")
        elif result == "not_found":
            b.speak("I can't see my dock. Please put it where I can see the marker.")
        elif result == "edge":
            b.speak("I'm facing a drop — turn me around and I'll try again.")
        elif result in {"no_contact", "contact", "alignment", "too_close", "posture"}:
            b.speak("I couldn't line up with my charger. Please help me onto the dock.")
        else:  # lost
            b.speak("I got confused on the way — I stopped somewhere safe. Can you carry me home?")
        return True

    # ----------------------------------------------------------------- voice
    def _voice_switch(self, name, server_voice, model, needs_module, pitch, robot_mix):
        b = self.body
        if model and not os.path.exists(model) and not server_voice:
            b.speak(f"I don't have the {name} voice files yet.")
            return True
        if needs_module and not os.path.exists(_MODULE_PY):
            b.speak(f"The {name} voice needs a speech upgrade I haven't got installed.")
            return True
        tts_cfg = self.cfg.setdefault("tts", {})
        if model:
            tts_cfg["piper_model"] = model
        else:
            tts_cfg.pop("piper_model", None)
        tts_cfg["voice_name"] = server_voice  # None = this voice is Pi-local only
        if needs_module:
            tts_cfg["piper_module_python"] = _MODULE_PY
        else:
            tts_cfg.pop("piper_module_python", None)
        tts_cfg["pitch_semitones"] = pitch
        tts_cfg["robot_mix"] = robot_mix
        self._persist_voice(model, needs_module, server_voice, pitch, robot_mix)
        # the confirmation itself speaks in the newly selected voice
        b.speak(f"This is my {name} voice now." if model
                else "Back to my original voice.")
        return True

    def _persist_voice(self, model, needs_module=False, server_voice=None,
                       pitch=0, robot_mix=0):
        """Best-effort: write the choice back to config.json so it survives
        restarts. The live switch already holds either way."""
        try:
            path = os.path.join(self.cfg.get("root", "/opt/spark/app"), "config.json")
            data = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            tts = data.setdefault("tts", {})
            if model:
                tts["piper_model"] = model
            else:
                tts.pop("piper_model", None)
            tts["voice_name"] = server_voice
            if needs_module:
                tts["piper_module_python"] = _MODULE_PY
            else:
                tts.pop("piper_module_python", None)
            tts["pitch_semitones"] = pitch
            tts["robot_mix"] = robot_mix
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _log(f"voice persist failed (live switch still holds): {e}")

    # ---------------------------------------------------------------- sched
    def _scheduled_fire(self, kind, label):
        """Alarm/timer/reminder went off — speak it (runs on its own thread)."""
        b = self.body
        b.eyes("thinking")
        if kind == "alarm":
            try:
                sfx = (self.cfg.get("sounds", {}).get("sfx_map", {}) or {}).get("alarm")
                if sfx:
                    b.play_sfx(sfx)
            except Exception:
                pass
            b.speak(f"Alarm! {label or 'Time to get moving!'}")
        elif kind == "reminder":
            b.speak(f"Reminder: {label or 'time is up'}!")
        else:
            b.speak(f"Time's up!{(' ' + label) if label else ''}")
        b.eyes("idle")

    def _alarms(self, text):
        """Set / cancel / query absolute-time alarms."""
        b = self.body
        if _ALARM_CANCEL_RE.search(text):
            n = self.alarms.cancel("alarm")
            b.speak("Alarm cancelled." if n else "You don't have any alarms set.")
            return True
        if _ALARM_QUERY_RE.search(text):
            lines = [l for l in self.alarms.status_lines() if l.startswith("alarm")]
            b.speak("You have " + "; ".join(lines) + "." if lines
                    else "You don't have any alarms set.")
            return True
        parsed = cmds.parse_clock_time(text)
        if not parsed:
            self._pending = {"kind": "alarm", "at": time.time()}
            b.speak("For what time should I set the alarm?")
            return True
        return self._set_alarm_from(parsed)

    def _set_alarm_from(self, parsed):
        hour, minute, mer = parsed
        when = next_occurrence(hour, minute, mer)
        if when is None:
            self.body.speak("I don't think that's a valid time.")
            return True
        self.alarms.add_alarm(when.timestamp(), label=fmt_clock(when.hour, when.minute))
        self.body.speak(f"Alarm set for {fmt_clock(when.hour, when.minute)}.")
        return True

    def _set_reminder(self, text):
        """'remind me to X in N minutes' -> labeled timer."""
        m = _REMINDER_PARSE_RE.search(text)
        if not m or not cmds.parse_timer(m.group(2)):
            self.body.speak("When should I remind you?")
            return True
        label = m.group(1).strip().strip(".,!")
        secs = cmds.parse_timer(m.group(2))
        self.alarms.add_timer(secs, label=label, kind="reminder")
        self.body.speak(f"Okay — I'll remind you to {label} in {self._describe_timer(secs).lower()}.")
        return True

    # --------------------------------------------------------------- timer
    def _start_timer(self, secs):
        t = threading.Timer(secs, self._timer_done, args=[secs])
        t.daemon = True
        t.start()
        self._timers.append(t)

    def _timer_done(self, secs):
        self.body.eyes("thinking")
        self.body.speak("Time's up!")
        self.body.eyes("idle")

    @staticmethod
    def _describe_timer(secs):
        if secs < 60:
            return f"Timer set for {int(secs)} seconds"
        if secs < 3600:
            m = secs / 60
            return f"Timer set for {int(m) if m == int(m) else round(m, 1)} minutes"
        return f"Timer set for {round(secs / 3600, 1)} hours"

    # --------------------------------------------------------------- imagine
    def _imagine(self, raw_text):
        """Narrate first, then perform once on the main SDK/recognition thread."""
        low = raw_text.lower()
        variant = next((val for key, val in _IMAGINE_THEMES.items() if key in low), "fiesta")
        if self.llm_reply is not None:
            self.llm_reply(
                raw_text,
                extra_context=("ROLEPLAY DIRECTION: Give one playful spoken sentence in character. "
                               "A robot routine will play AFTER your sentence. Do not claim to "
                               "navigate to a person or place. Do not include stage directions."),
            )
        else:
            self.body.speak("Let's pretend. Watch this!")
        _log(f"imagine: variant={variant}")
        if not self.body.dance(variant):
            self.body.speak("My routine didn't finish that time.")
        self.body.eyes("idle")
        return True

    # ---------------------------------------------------------------- search
    def _web_search(self, raw_text):
        """Search the web, then let the brain answer from the results."""
        query = _SEARCH_INTENT.sub(" ", raw_text)
        query = cmds.normalize(query) or cmds.normalize(raw_text)
        self.body.eyes("thinking")
        self.body.speak("Let me look that up.")
        results = websearch.web_search(query, max_results=4)
        if not results:
            self.body.speak("I couldn't reach the web for that one.")
            self.body.eyes("idle")
            self.memory.add("user", raw_text)
            return
        _log(f"search '{query}': {len(results)} results")
        context = websearch.context_block(query, results)
        if self.llm_reply is not None:
            # results already fetched; one READ hop so she can open a page
            self.llm_reply(raw_text, extra_context=context, web_hops=1)
        else:
            for r in results[:2]:
                self.body.speak(f"{r['title']}. {r['snippet']}")
        self.body.eyes("idle")

    # --------------------------------------------------------------- weather
    def _weather(self):
        try:
            loc = self.cfg["location"]
            lat, lon = loc.get("lat"), loc.get("lon")
            if lat is None or lon is None:
                geo = json.loads(urllib.request.urlopen(
                    "http://ip-api.com/json/", timeout=4).read())
                lat, lon = geo["lat"], geo["lon"]
            url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={lat}&longitude={lon}&temperature_unit=fahrenheit"
                   f"&wind_speed_unit=mph&current=temperature_2m,"
                   f"apparent_temperature,weather_code,wind_speed_10m")
            data = json.loads(urllib.request.urlopen(url, timeout=5).read())["current"]
            desc = self._wmo(data.get("weather_code"))
            return (f"Right now it's {int(data['temperature_2m'])} degrees, "
                    f"feels like {int(data['apparent_temperature'])}. {desc}.")
        except Exception as e:
            _log(f"weather failed: {e}")
            return None

    @staticmethod
    def _wmo(code):
        mapping = {0: "Clear skies", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
                   45: "Foggy", 51: "Light drizzle", 61: "Light rain", 63: "Rain",
                   65: "Heavy rain", 71: "Snow", 80: "Rain showers", 95: "Thunderstorms"}
        return mapping.get(code, f"Weather code {code}")
