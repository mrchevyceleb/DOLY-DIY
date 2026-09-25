"""Locally supported stock-style voice commands.

Matching: normalized fuzzy match over phrase lists. Extend `COMMANDS`
with anything you find in the Doly app (Main page → Interact → Say).
"""
import difflib
import re
import threading
import time

COLORS = ["Red", "Green", "Blue", "White", "Yellow", "Orange", "Purple", "Pink",
          "Cyan", "Magenta", "DarkGreen", "LightBlue", "Black"]

WAKE_WORDS = ["hey spark", "spark", "hey sparky", "sparky", "hey doly", "dolly", "doly", "hey dolly"]

# command -> (phrases, params).  action names map to Body methods / handlers.
COMMANDS = [
    {"action": "time", "phrases": ["what time is it", "what's the time", "tell me the time",
                                    "the time please"]},
    {"action": "date", "phrases": ["what's the date", "what day is it", "what's today"]},
    {"action": "timer", "phrases": ["set a timer", "set timer", "timer for", "set a timer for"]},
    {"action": "weather", "phrases": ["how is the weather", "what's the weather", "weather today",
                                       "how's the weather", "is it going to rain"]},
    {"action": "photo", "phrases": ["take a photo", "take a picture", "say cheese",
                                     "take my picture"]},
    {"action": "fist_bump", "phrases": ["fist bump", "give me a fist bump", "bump it"]},
    {"action": "high_five", "phrases": ["high five", "give me a five", "give me five",
                                        "give me a high five", "gimme five", "up top"]},
    {"action": "go_home", "phrases": ["go home", "return home", "go back home", "head home",
                                      "go back to your dock", "go to your dock", "back to your dock",
                                      "go back to the dock", "go to the dock", "back to the dock",
                                      "return to the dock", "head to the dock", "drive to the dock",
                                      "go to dock", "back to dock", "return to dock",
                                      "get on the dock", "get on your dock", "park yourself",
                                      "go back to the charger", "back to the charger",
                                      "return to the charger", "head to the charger",
                                      "dock yourself", "go charge", "go and charge", "go to bed",
                                      "go to your charger", "back to your charger", "go charge yourself",
                                      "return to your charger", "charge up", "go plug in", "get home",
                                      "go to the charger", "time to charge", "go recharge"]},
    {"action": "dance_salsa", "phrases": ["salsa", "fiesta dance"]},
    {"action": "dance_twist", "phrases": ["do the twist", "twist dance"]},
    {"action": "dance_rock", "phrases": ["rock dance", "rock and roll", "let's rock"]},
    {"action": "dance_workout", "phrases": ["work out", "workout", "do exercise", "do some exercise"]},
    {"action": "dance_party", "phrases": ["party dance", "happy dance", "today is my birthday"]},
    {"action": "dance_meditate", "phrases": ["meditate", "meditation"]},
    {"action": "dance", "phrases": ["dance", "do a dance", "let's dance", "do a little dance",
                                     "show me your moves"]},
    {"action": "come_here", "phrases": ["come here", "come to me", "come over here",
                                         "come to me please"]},
    {"action": "spin", "phrases": ["spin", "spin around", "do a spin", "turn around"]},
    {"action": "forward", "phrases": ["go forward", "move forward", "drive forward", "forward",
                                     "move forwards", "go forwards", "forwards", "move ahead"]},
    {"action": "back", "phrases": ["go back", "move back", "back up", "drive backward",
                                    "backwards", "backward", "move backward", "move backwards",
                                    "go backwards", "reverse", "back up now"]},
    {"action": "left", "phrases": ["turn left", "go left", "move left"]},
    {"action": "right", "phrases": ["turn right", "go right", "move right"]},
    {"action": "stop", "phrases": ["stop", "stop moving", "halt"]},
    {"action": "sleep", "phrases": ["go to sleep", "sleep", "sleep mode", "goodnight",
                                     "good night"]},
    {"action": "wake", "phrases": ["wake up", "good morning spark", "wakey wakey"]},
    {"action": "battery", "phrases": ["how's your battery", "what's your battery level",
                                       "are you charged", "how much battery"]},
]

MATCH_THRESHOLD = 0.72
# Motion and other consequential actions need stronger confidence, and
# never fire from negated or embedded speech ("don't dance", "back to the future").
MOTION_ACTIONS = {"dance", "come_here", "spin", "forward", "back", "left", "right",
                  "stop", "fist_bump", "high_five", "sleep", "photo", "go_home"}
MOTION_ACTIONS.update(c["action"] for c in COMMANDS if c["action"].startswith("dance_"))
MOTION_THRESHOLD = 0.82

# politeness/filler tokens stripped before matching ("please dance" -> "dance")
_FILLER = {"please", "can", "you", "could", "would", "will", "just", "now",
           "hey", "okay", "ok", "a", "the", "me", "for"}

_NEGATION_RE = re.compile(r"\b(don'?t|dont|didn'?t|didnt|doesn'?t|isn'?t|wasn'?t|"
                          r"can'?t|won'?t|not|never|no|stop (?:talking|asking))\b")
_DOCK_CORRECTION_RE = re.compile(
    r"^(?:no )?you(?:'re| are) not (?:on|at) (?:a |the |your )?"
    r"(?:charging dock|charger|dock)\b\s*")

_norm_re = re.compile(r"[^a-z0-9' ]+")


def normalize(text):
    text = (text or "").lower().strip()
    text = _norm_re.sub(" ", text)
    text = " ".join(text.split())
    text = re.sub(r"^(?:okay|ok) (?=(?:spark|sparky|doly|dolly)\b)", "", text)
    changed = True
    while changed:
        changed = False
        for w in WAKE_WORDS:
            if text.startswith(w + " "):
                text = text[len(w) + 1:]
                changed = True
            if text == w:
                return ""
    return text


def _tokens(s):
    return s.split()


def _token_subseq(text_tokens, phrase_tokens):
    """True if phrase tokens appear contiguously, word-aligned, in text."""
    n, m = len(text_tokens), len(phrase_tokens)
    if m == 0 or m > n:
        return False
    return any(text_tokens[i:i + m] == phrase_tokens for i in range(n - m + 1))


def _score(text, phrase):
    if text == phrase:
        return 1.0
    tt, pt = _tokens(text), _tokens(phrase)
    if _token_subseq(tt, pt):
        # word-aligned command inside a longer sentence: scale by coverage
        return 0.55 + 0.45 * (len(pt) / len(tt))
    return difflib.SequenceMatcher(None, text, phrase).ratio()


_STOP_CMD = {"action": "stop", "phrases": ["stop"]}


def match_command(text):
    """Only imperatives authorize motion; a mention or complaint does not."""
    # Keep sentence boundaries: a complaint can precede a fresh request.
    clauses = [c for c in re.split(r"[.!?;]+", text) if c.strip()]
    if len(clauses) > 1:
        prohibited = any(re.match(r"^(?:please )?(?:don'?t|do not|never|no)\b",
                                 _DOCK_CORRECTION_RE.sub("", normalize(c))) for c in clauses)
        for clause in reversed(clauses):
            cmd, score = match_command(clause)
            if cmd:
                if prohibited and cmd["action"] in MOTION_ACTIONS and cmd["action"] != "stop":
                    return None, 0.0
                return cmd, score
        return None, 0.0
    text = normalize(text)
    # A correction of the robot's dock claim does not negate the following
    # imperative: "No, you're not on a charger. Come here." Keep all other
    # negations, including "...don't come here", intact.
    text = _DOCK_CORRECTION_RE.sub("", text)
    text = re.sub(r"^(?:(?:okay|ok|hey|please) )*why don'?t you ", "", text)
    if not text:
        return None, 0.0
    tokens = _tokens(text)

    # emergency stop wins: leading/trailing bare 'stop' always stops the robot
    if tokens and (tokens[0] == "stop" or tokens[-1] == "stop"):
        return _STOP_CMD, 0.9

    negated = bool(_NEGATION_RE.search(text))
    best, best_score = None, 0.0
    for cmd in COMMANDS:
        is_motion = cmd["action"] in MOTION_ACTIONS
        for phrase in cmd["phrases"]:
            pt = _tokens(phrase)
            n = len(pt)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] != pt:
                    continue
                s = 1.0
                prefix = " ".join(tokens[:i])
                request_prefix = re.fullmatch(
                    r"(?:(?:okay|ok|hey|please|just|now) )*"
                    r"(?:(?:can|could|would|will) you |i want you to |"
                    r"i would like you to |let's |lets |give me |show me |do )?"
                    r"(?:(?:please|just|a|the|another) )*", prefix + " " if prefix else "")
                if is_motion and (negated or not request_prefix):
                    s = 0.5  # negated speech can never trigger motion
                if s >= (MOTION_THRESHOLD if is_motion else MATCH_THRESHOLD) and s > best_score:
                    best, best_score = cmd, s
    if best is not None:
        return best, best_score

    # fuzzy fallback (non-motion only): whole-string similarity
    match_text = " ".join(t for t in tokens if t not in _FILLER) or text
    for cmd in COMMANDS:
        if cmd["action"] in MOTION_ACTIONS:
            continue
        for phrase in cmd["phrases"]:
            s = difflib.SequenceMatcher(None, match_text, phrase).ratio()
            if s >= MATCH_THRESHOLD and s > best_score:
                best, best_score = cmd, s
    return best, best_score


def extract_color(text):
    low = text.lower()
    for c in COLORS:
        if c.lower() in low:
            return c
    return None


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "a": 1, "an": 1, "half": 0.5,
}


def parse_timer(text):
    """'set a timer for 5 minutes' / 'five minutes' -> seconds (or None)."""
    low = text.lower()
    # Vosk often emits spoken numbers as words — normalize first
    for word, val in sorted(_NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        low = re.sub(rf"\b{word}\b", str(val), low)
    # collapse compounds: 'twenty 5' -> 25 (tens + digit)
    low = re.sub(r"\b(\d+)\s+(\d)\b", lambda m: str(int(m.group(1)) + int(m.group(2))), low)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(second|seconds|minute|minutes|min|hour|hours|hrs?)", low)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2)
    if unit.startswith("second"):
        return n
    if unit.startswith("min"):
        return n * 60
    return n * 3600
