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


def _log(msg):
    print(f"[router] {msg}", file=sys.stderr)


# "Imagine..." -> she acts it out: themed routine + the brain narrating theatrically
_IMAGINE_RE = re.compile(r"(imagine|pretend|act like|act out)", re.IGNORECASE)
_IMAGINE_THEMES = {
    "beach": ("fiesta", "/.doly/sounds/sfx/beach.wav"),
    "party": ("party", None),
    "exercise": ("groove", "/.doly/sounds/sfx/buff (1).wav"),
    "workout": ("groove", "/.doly/sounds/sfx/buff (2).wav"),
    "birthday": ("party", "/.doly/sounds/music/birthday.wav"),
    "dance": ("fiesta", None),
    "robot": ("groove", None),
}

# Explicit web-search intent — always goes to the search tool, not the stock table.
_SEARCH_INTENT = re.compile(
    r"\b(?:search (?:the web |the internet )?(?:for |about )?|look up|google|"
    r"what'?s the latest (?:on |with )?|any news (?:on |about )?|news about)\b",
    re.IGNORECASE,
)


EDGE_REFUSAL = ("I can't drive here — I'm either on my dock or too close to an edge. "
                "Put me somewhere with room and ask again!")
DOCK_REFUSAL = "My wheels don't reach down here — I'm on my charging dock! Lift me onto the desk and I'll scoot."

# Voice switching by name — aliases include common ASR mishearings
# ("switch to weekly" really is how "wheatley" comes back from the mic).
# alias -> (display name, model path|None, needs modern piper module)
_MODULE_PY = "/opt/piper-ng/bin/python"
_VOICES = {
    "wheatley": ("Wheatley", "/.doly/data/piper/wheatley-en.onnx", True),
    "weekly": ("Wheatley", "/.doly/data/piper/wheatley-en.onnx", True),
    "glados": ("GLaDOS", "/.doly/data/piper/glados.onnx", False),
    "gladys": ("GLaDOS", "/.doly/data/piper/glados.onnx", False),
    "amy": ("Amy", "/.doly/data/piper/en_US-amy-medium.onnx", False),
    "hfc": ("HFC", "/.doly/data/piper/en_US-hfc_female-medium.onnx", False),
    "lessac": ("Lessac", "/.doly/data/piper/en_US-lessac-high.onnx", False),
    "stock": ("stock", None, False),
    "original": ("stock", None, False),
}
_SWITCH_VERB_RE = re.compile(r"\b(switch|change|swap|use|try)\b", re.IGNORECASE)
_VOICE_LIST_RE = re.compile(r"\b(what|which|list)\b", re.IGNORECASE)


def _voice_intent(low):
    """('display name', model path|None) when the text asks to switch voices."""
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
        self.llm_reply = None  # set by Spark: streamed brain reply w/ context
        self._timers = []

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

        # imagine prompts: she physically acts it out while narrating
        if _IMAGINE_RE.search(raw_text):
            return self._imagine(raw_text)

        # explicit web search → tool + brain-mediated answer
        if _SEARCH_INTENT.search(raw_text):
            self._web_search(raw_text)
            return True

        # color commands (need param extraction before fuzzy match)
        low = text.lower()
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

        # voice management: "switch to wheatley", "what voices do you have"
        voice = _voice_intent(low)
        if voice is not None:
            name, model, needs_module = voice
            return self._voice_switch(name, model, needs_module)
        if "voice" in low and _VOICE_LIST_RE.search(low):
            current = self.cfg.get("tts", {}).get("piper_model")
            current = os.path.basename(current).split(".")[0] if current else "stock"
            b_names = ", ".join(sorted({n for n, _, _ in _VOICES.values()}))
            self.body.speak(f"I'm using the {current} voice. I can also be: {b_names}. "
                            "Just say switch to, and a name.")
            return True

        cmd, score = cmds.match_command(text)
        if cmd:
            _log(f"command={cmd['action']} score={score:.2f} text='{text}'")
            return self._execute(cmd["action"], text)

        return False

    # -------------------------------------------------------------- executor
    def _execute(self, action, text):
        b = self.body

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
                b.speak("How long should I set it for?")
                return True
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
            b.speak("Bump!")
            b.fist_bump()
            return True

        if action == "high_five":
            b.speak("Up top!")
            b.high_five()
            return True

        if action == "go_home":
            return self.go_home_action()

        if action == "dance":
            b.speak("Watch this.")
            if not b.dance():
                # never a flat refusal: always perform SOMETHING
                b.speak("No room to spin here — arms party!")
                b.arms_party()
            return True

        if action == "come_here":
            b.speak("On my way.")
            if not b.drive_distance(250):
                b.speak(EDGE_REFUSAL)
            return True

        if action == "spin":
            b.speak("Wheee.")
            if not b.drive_rotate(360):
                b.speak(EDGE_REFUSAL)
            return True

        def _refuse():
            b.speak(DOCK_REFUSAL if getattr(b, "docked", False) else EDGE_REFUSAL)

        if action == "forward":
            if not b.drive_distance(150):
                _refuse()
            return True
        if action == "back":
            if not b.drive_distance(-150):
                _refuse()
            return True
        if action == "left":
            if not b.drive_rotate(-90):
                _refuse()
            return True
        if action == "right":
            if not b.drive_rotate(90):
                _refuse()
            return True
        if action == "stop":
            b.stop_everything()
            return True

        if action == "sleep":
            b.speak("Powering down. Tap me when you need me.")
            b.sleep_pose()
            return True

        if action == "wake":
            b.speak("Morning! Fully charged and ready.")
            b.wake_up()
            return True

        if action == "battery":
            pct = b.battery_pct()
            if pct is None:
                b.speak("I can't read my battery right now.")
            else:
                b.speak(f"I'm at {pct} percent.")
            return True

        _log(f"unhandled action {action}")
        return False

    # ----------------------------------------------------------------- homing
    def go_home_action(self):
        b = self.body
        result = b.go_home()
        if result == "already":
            b.speak("I'm already home, all cozy.")
        elif result == "arrived":
            b.speak("Home sweet home. Charging up!")
        elif result == "unknown":
            b.speak("I don't know where home is right now — put me on my dock once and I'll remember it.")
        elif result == "busy":
            b.speak("I'm already heading home!")
        else:  # lost
            b.speak("I got confused on the way — I stopped somewhere safe. Can you carry me home?")
        return True

    # ----------------------------------------------------------------- voice
    def _voice_switch(self, name, model, needs_module):
        b = self.body
        if model and not os.path.exists(model):
            b.speak(f"I don't have the {name} voice files yet.")
            return True
        if needs_module and not os.path.exists(_MODULE_PY):
            b.speak(f"The {name} voice needs a speech upgrade I haven't got installed.")
            return True
        tts_cfg = self.cfg.setdefault("tts", {})
        if model:
            tts_cfg["piper_model"] = model
            if needs_module:
                tts_cfg["piper_module_python"] = _MODULE_PY
            else:
                tts_cfg.pop("piper_module_python", None)
        else:
            tts_cfg.pop("piper_model", None)
            tts_cfg.pop("piper_module_python", None)
        self._persist_voice(model, needs_module)
        # the confirmation itself speaks in the newly selected voice
        b.speak(f"This is my {name} voice now." if model
                else "Back to my original voice.")
        return True

    def _persist_voice(self, model, needs_module=False):
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
                if needs_module:
                    tts["piper_module_python"] = _MODULE_PY
                else:
                    tts.pop("piper_module_python", None)
            else:
                tts.pop("piper_model", None)
                tts.pop("piper_module_python", None)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            _log(f"voice persist failed (live switch still holds): {e}")

    # ----------------------------------------------------------------- timer
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
        """Stock's best bit, upgraded: themed physical routine + LLM narration."""
        low = raw_text.lower()
        variant, theme_sfx = "fiesta", None
        for key, val in _IMAGINE_THEMES.items():
            if key in low:
                variant, theme_sfx = val
                break
        import threading

        def _perform():
            try:
                self.body.eyes("speaking")
                if theme_sfx:
                    self.body.play_sfx(theme_sfx, defer=False)
                self.body.dance(variant)
                self.body.eyes("idle")
            except Exception as e:
                _log(f"imagine perform failed: {e}")

        # she performs on a side stage while the brain narrates the fantasy
        threading.Thread(target=_perform, daemon=True).start()
        _log(f"imagine: variant={variant}")
        if self.llm_reply is not None:
            self.llm_reply(
                raw_text,
                extra_context=("ROLEPLAY DIRECTION: You are physically acting this out RIGHT NOW — "
                               "arms waving, wheels spinning, in character. Narrate it with "
                               "theatrical flair, present tense, like a tiny robot living its "
                               "best fantasy. No stage directions in brackets — just spoken words."),
            )
        else:
            self.body.speak("Oh, I love this one. Watch me!")
            self.body.dance(variant)
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
            self.llm_reply(raw_text, extra_context=context)
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
                   f"latitude={lat}&longitude={lon}&current=temperature_2m,"
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
