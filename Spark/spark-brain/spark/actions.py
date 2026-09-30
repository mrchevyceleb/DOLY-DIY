"""Compound commands: several things in one breath, or a mood for the room.

'Dim the lights and wake me at seven.' 'Make it cozy.' The single-command
routes act on the first thing they recognise and drop the rest, so these
go to the brain for a plan instead: one action per line from a short fixed
menu. This code runs the plan and says what actually happened; the brain
never speaks the confirmation, so it cannot claim something that failed.

Movement is never planned by the brain. A clause that is a movement
command on its own ('... and go home') is matched by the same strict
matcher as any other turn and runs last.
"""
import datetime
import re
import sys
import threading
import time

from . import commands as cmds

_SPLIT_RE = re.compile(r"\s*(?:[,;.!?]|\band then\b|\band also\b|\band\b|\bthen\b|\balso\b|\bplus\b)\s*",
                       re.I)
# 'what alarms and timers do I have?' is a question, not two orders
_QUESTION_RE = re.compile(r"^(?:what|which|when|where|who|why|how|do i|did|does|is|are|was|were)\b")
_DOMAINS = (
    re.compile(r"\b(?:govee|lights?|lamps?)\b", re.I),
    re.compile(r"\btimer\b", re.I),
    re.compile(r"\balarms?\b|\bwake me\b", re.I),
    re.compile(r"\bremind me\b", re.I),
    re.compile(r"\b(?:remember that|make a note|don'?t forget)\b", re.I),
)
# an order that names nothing itself ('... and make them warm')
_VERB_RE = re.compile(r"(?:(?:please|just|now) )*(?:make|set|turn|put|change|dim|brighten|"
                      r"switch|cancel|wake|remind|start|add)\b", re.I)
_HERS_RE = re.compile(r"\byour\s+(?:lights?|leds?|eyes?)\b", re.I)
# a mood for the room, asked for rather than mentioned
_MOOD_RE = re.compile(
    r"\b(?:make|get) (?:it|the room|this room|things)"
    r"(?: nice and| a (?:bit|little)| more| really)? "
    r"(?:cozy|cosy|cozier|bright|brighter|dark|darker|warm|warmer|relaxing|romantic|calm|chill)\b|"
    r"\bset the mood\b|\bget (?:it|things|the room) ready for\b|"
    r"\b(?:movie|cozy|cosy|focus|reading|bedtime|relax|chill) (?:mode|time|night|lighting|lights)\b|"
    r"\bit'?s too (?:bright|dark)\b", re.I)

_MENU = """You turn Matt's spoken request into actions for his desk robot Spark. \
Reply with one action per line and nothing else. The only actions:
lights on
lights off
lights brightness <1-100>
lights color <red|orange|gold|yellow|green|teal|cyan|blue|purple|pink|magenta|white>
lights temp <kelvin>        2700 is warm and cozy, 4000 neutral, 6500 daylight
lights scene <name>         a lamp scene, only when he names it: aurora, sunset glow, candlelight, movie, reading, party
timer <seconds>
alarm <H:MM> [am|pm]        add am or pm only when he said it or it is obvious
reminder <seconds> <what to remind him of>
note <a fact he asked you to remember>
cancel timer
cancel alarm
Plan only what he asked for, in the order he said it. A mood ('cozy', 'movie \
night', 'time to focus') means the room lights: pick a brightness and a temp or \
color that fit. Moving, dancing, going home, sleeping and questions are handled \
elsewhere: leave them out. If nothing on the list fits, reply with the one word: none"""


def _log(msg):
    print(f"[actions] {msg}", file=sys.stderr)


def _clauses(raw):
    return [c for c in _SPLIT_RE.split(raw or "") if c.strip()]


def _movement(clause):
    """The stock movement command a clause gives by itself, or None."""
    cmd, _ = cmds.match_command(clause)
    if cmd and cmd["action"] in cmds.MOTION_ACTIONS and cmd["action"] != "stop":
        return cmd["action"]
    return None


def _actionable(clause):
    if _HERS_RE.search(clause):
        return False   # her own LEDs and eyes are single commands
    return (any(rx.search(clause) for rx in _DOMAINS) or bool(_MOOD_RE.search(clause))
            or _movement(clause) is not None)


def wanted(raw):
    """True for two or more orders in one sentence, or a mood for the room."""
    text = cmds.normalize(raw)
    if not text or _QUESTION_RE.match(text):
        return False
    if _MOOD_RE.search(raw) and not cmds._NEGATION_RE.search(text):
        return True
    clauses = _clauses(raw)
    if sum(1 for c in clauses if _actionable(c)) >= 2:
        return True
    # 'dim the lights and make them warm': the second order leans on the first
    named = [c for c in clauses if any(rx.search(c) for rx in _DOMAINS) and not _HERS_RE.search(c)]
    return len(named) == 1 and any(_VERB_RE.match(c.strip()) and not _HERS_RE.search(c)
                                   for c in clauses if c is not named[0])


def _ask(brain, raw, looks=(), timeout=10.0):
    """The brain's plan as text; None when it cannot be reached in time."""
    box = {}
    menu = _MENU
    if looks:
        menu += ("\nHis saved looks, when he asks for one by name: "
                 + ", ".join(f"lights look {n}" for n in looks))
    messages = [
        {"role": "system", "content": menu},
        {"role": "user", "content": f"It is {datetime.datetime.now():%A %I:%M %p}. "
                                    f"Matt said: {raw.strip()}"},
    ]

    def ask():
        try:
            box["out"] = brain.chat(messages, fallback=False, temperature=0.1, max_tokens=120)
        except Exception as e:
            box["err"] = e
    t = threading.Thread(target=ask, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive() or "err" in box:
        _log(f"plan failed: {box.get('err', 'timeout')}")
        return None
    return box.get("out") or ""


def _apply(router, plan):
    """Run each planned line; returns the sentences to say."""
    said = []
    g = router.govee if getattr(router.govee, "enabled", False) else None
    alarms = getattr(router, "alarms", None)
    lights_down = clock_down = False
    for line in plan.splitlines()[:8]:
        line = line.strip().strip("-*`•").strip()
        m = re.match(r"(lights|timer|alarm|reminder|note|cancel)\b\s*(.*)$", line, re.I)
        if not m:
            continue
        verb, rest = m.group(1).lower(), m.group(2).strip()
        _log(f"{verb} {rest}")
        try:
            if verb == "lights":
                if g is None:
                    lights_down = True
                    continue
                op = re.match(r"(on|off|brightness|color|colour|temp|scene|look)\b\s*(.*)$", rest, re.I)
                if not op:
                    continue
                kind, arg = op.group(1).lower(), op.group(2).strip().rstrip(".")
                num = re.search(r"\d+", arg)
                if kind in ("on", "off"):
                    said.append(g.turn(kind == "on"))
                elif kind == "brightness" and num:
                    said.append(g.brightness(int(num.group())))
                elif kind == "temp" and num:
                    said.append(g.color_temp(int(num.group())))
                elif kind in ("color", "colour") and arg:
                    said.append(g.color(arg))
                elif kind == "scene" and arg:
                    said.append(g.scene(arg))
                elif kind == "look" and arg:
                    said.append(router.light_shortcut("", name=arg.lower()))
            elif verb == "note":
                pet = getattr(router, "pet", None)
                if pet is not None and len(rest.split()) >= 2:
                    pet._add_note(f"Matt said: {rest}")
                    said.append("I'll remember that.")
            elif alarms is None:
                clock_down = True
            elif verb == "timer":
                num = re.match(r"(\d+)", rest)
                if num and 1 <= int(num.group(1)) <= 86400:
                    secs = int(num.group(1))
                    alarms.add_timer(secs)
                    router._last_set = {"kind": "timer", "at": time.time()}
                    said.append(router._describe_timer(secs) + ".")
            elif verb == "reminder":
                num = re.match(r"(\d+)\s+(.+)$", rest)
                if num and 1 <= int(num.group(1)) <= 86400:
                    secs, label = int(num.group(1)), num.group(2).strip(" .")[:120]
                    label = re.sub(r"^to\s+", "", label, flags=re.I)
                    alarms.add_timer(secs, label=label, kind="reminder")
                    span = router._describe_timer(secs)[len("Timer set for "):]
                    said.append(f"I'll remind you to {label} in {span}.")
            elif verb == "alarm":
                at = re.match(r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", rest, re.I)
                if at:
                    said.append(router._arm_alarm(
                        (int(at.group(1)), int(at.group(2) or 0), at.group(3))))
            elif verb == "cancel":
                kind = "alarm" if "alarm" in rest.lower() else "timer" if "timer" in rest.lower() else None
                if kind == "alarm":
                    router._stop_sunrise()
                if kind:
                    n = alarms.cancel(kind)
                    said.append(f"{kind.capitalize()} cancelled." if n
                                else f"There's no {kind} to cancel.")
        except Exception as e:
            _log(f"'{line}' failed: {e}")
            said.append("One of those didn't work.")
    if lights_down:
        said.append("I can't reach the lights right now.")
    if clock_down:
        said.append("My clock isn't available right now.")
    # two light changes that both failed say so once
    return list(dict.fromkeys(s for s in said if s))


def run(router, raw):
    """Plan and carry out a compound request. False: nothing was planned,
    so the caller routes the sentence the ordinary way."""
    with router.body.busy("thinking"):
        plan = _ask(router.brain, raw, sorted((router.cfg.get("govee", {}) or {}).get("shortcuts") or {}))
    said = _apply(router, plan) if plan else []
    if not said:
        router.body.eyes("idle")
        return False
    router.body.speak(" ".join(said))
    router.body.eyes("idle")
    # one movement order may ride along; it keeps its own guards and lines
    for clause in _clauses(raw):
        action = _movement(clause)
        if action:
            _log(f"movement clause: {action}")
            router._execute(action, cmds.normalize(clause))
            break
    return True
