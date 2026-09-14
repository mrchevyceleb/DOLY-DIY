"""Router — stock commands first, LLM for everything else.

Phase 1 policy:
- A stock command match executes locally (instant, works offline).
- Anything else goes to the brain on Moria.
- If the brain is offline, free speech degrades gracefully.
"""
import datetime
import json
import re
import sys
import threading
import time
import urllib.request

from . import commands as cmds
from . import search as websearch


def _log(msg):
    print(f"[router] {msg}", file=sys.stderr)


# Explicit web-search intent — always goes to the search tool, not the stock table.
_SEARCH_INTENT = re.compile(
    r"\b(?:search (?:the web |the internet )?(?:for |about )?|look up|google|"
    r"what'?s the latest (?:on |with )?|any news (?:on |about )?|news about)\b",
    re.IGNORECASE,
)


EDGE_REFUSAL = "Whoa, nope, that's too close to the edge. Not falling for that again."


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
            self.body.drive_stop()
            return True

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

        if action == "dance":
            b.speak("Watch this.")
            if not b.dance():
                b.speak(EDGE_REFUSAL)
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

        if action == "forward":
            if not b.drive_distance(150):
                b.speak(EDGE_REFUSAL)
            return True
        if action == "back":
            if not b.drive_distance(-150):
                b.speak(EDGE_REFUSAL)
            return True
        if action == "left":
            if not b.drive_rotate(-90):
                b.speak(EDGE_REFUSAL)
            return True
        if action == "right":
            if not b.drive_rotate(90):
                b.speak(EDGE_REFUSAL)
            return True
        if action == "stop":
            b.drive_stop()
            return True

        if action == "sleep":
            b.speak("Powering down. Tap me when you need me.")
            b.sleep_pose()
            return True

        if action == "wake":
            b.speak("Morning! Fully charged and ready.")
            b.eyes("idle")
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
