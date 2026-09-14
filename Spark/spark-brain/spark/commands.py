"""Stock command parity — every stock-style voice command, kept working.

Matching: normalized fuzzy match over phrase lists. Extend `COMMANDS`
with anything you find in the Doly app (Main page → Interact → Say).
"""
import difflib
import re
import threading
import time

COLORS = ["Red", "Green", "Blue", "White", "Yellow", "Orange", "Purple", "Pink",
          "Cyan", "Magenta", "DarkGreen", "LightBlue", "Black"]

WAKE_WORDS = ["hey spark", "spark", "hey doly", "dolly", "doly", "hey dolly"]

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
    {"action": "fist_bump", "phrases": ["fist bump", "give me a fist bump", "bump it",
                                      "give me a five", "high five", "give me five"]},
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
    {"action": "left", "phrases": ["turn left", "go left"]},
    {"action": "right", "phrases": ["turn right", "go right"]},
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
                  "stop", "fist_bump", "sleep", "photo"}
MOTION_THRESHOLD = 0.82

# politeness/filler tokens stripped before matching ("please dance" -> "dance")
_FILLER = {"please", "can", "you", "could", "would", "will", "just", "now",
           "hey", "okay", "ok", "a", "the", "me", "for"}

_NEGATION_RE = re.compile(r"\b(don'?t|dont|do not|never|no|stop (?:talking|asking))\b")

_norm_re = re.compile(r"[^a-z0-9' ]+")


def normalize(text):
    text = (text or "").lower().strip()
    text = _norm_re.sub(" ", text)
    text = " ".join(text.split())
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
    """Return the best-matching command dict + score, or (None, 0)."""
    text = normalize(text)
    if not text:
        return None, 0.0
    tokens = _tokens(text)

    # emergency stop wins: leading/trailing bare 'stop' always stops the robot
    if tokens and (tokens[0] == "stop" or tokens[-1] == "stop"):
        return _STOP_CMD, 0.9

    negated = bool(_NEGATION_RE.search(text))
    # strip politeness tokens for matching (keeps original for logging)
    match_text = " ".join(t for t in _tokens(text) if t not in _FILLER) or text
    best, best_score = None, 0.0
    for cmd in COMMANDS:
        threshold = MOTION_THRESHOLD if cmd["action"] in MOTION_ACTIONS else MATCH_THRESHOLD
        for phrase in cmd["phrases"]:
            # single-word motion phrases demand an exact word match —
            # character-fuzz would accept 'spine'->spin, 'sleepy'->sleep
            if len(_tokens(phrase)) == 1 and cmd["action"] in MOTION_ACTIONS:
                if match_text == phrase or _token_subseq(_tokens(match_text), _tokens(phrase)):
                    s = 1.0 if match_text == phrase else 0.55 + 0.45 / len(_tokens(match_text))
                else:
                    continue
            else:
                s = _score(match_text, phrase)
            if negated and cmd["action"] in MOTION_ACTIONS:
                s = min(s, 0.5)  # negated speech can never trigger motion
            if s >= threshold and s > best_score:
                best, best_score = cmd, s
    if best is not None:
        return best, best_score
    return None, best_score


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
