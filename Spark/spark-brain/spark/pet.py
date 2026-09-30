"""Pet life: what makes Spark a companion instead of a speaker.

She greets Matt when he comes back, answers praise with happy sounds
instead of sentences, gets drowsy at night and bored when ignored, learns
tricks, plays games, keeps notes about his life, and now and then says
something on her own. Everything runs on the main thread except note
extraction, the weather peek and the lamp pulse (HTTP only, no audio).

Rules carried over from the rest of her body: no random noises (every
sound here answers Matt or is a rare, deliberate remark), no wheel motion
except the explicit spin/chase tricks through the guarded turn, and the
brain is only ever asked with the resident model (no fallback).
"""
import datetime
import json
import pathlib
import random
import re
import threading
import time

SOUND = "/.doly/sounds/sound"
SFX = "/.doly/sounds/sfx"
ANIMAL = "/.doly/sounds/animal"
MUSIC = "/.doly/sounds/music"
_COUNTS = {"admire": 22, "agree": 15, "angry": 26, "call": 9, "cry": 5, "die": 9,
           "happy": 4, "laugh": 22, "sad": 20, "shock": 12, "sneeze": 7, "yawn": 4}

AWAY_S = 15 * 60        # an absence this long earns a welcome back
LOOK_EVERY_S = 3 * 60   # camera presence check cadence while Matt is quiet
QUIET_BEFORE_LOOK_S = 5 * 60
NOTICE_GAP_S = 45 * 60  # unprompted remarks: at most one per 45 minutes
ANSWER_WINDOW_S = 45


def _log(msg):
    print(f"[pet] {msg}", flush=True)


def stock(kind):
    """Random take of a stock Doly voice sound ('happy' -> happy (3).wav)."""
    return f"{SOUND}/{kind} ({random.randint(1, _COUNTS[kind])}).wav"


def _in_window(now, start, end):
    """now: datetime; start/end 'HH:MM'. Windows may wrap midnight."""
    def mins(s):
        h, m = str(s).split(":")
        return int(h) * 60 + int(m)
    cur, s, e = now.hour * 60 + now.minute, mins(start), mins(end)
    return s <= cur < e if s < e else (cur >= s or cur < e)


# ------------------------------------------------------------------ phrases
_NAME = r"(?:spark|sparky|buddy|girl)"
_PRAISE_RE = re.compile(
    r"^(?:(?:oh|aw+|aww+|hey|wow)\s+)?(?:" + _NAME + r"\s+)?(?:"
    r"(?:what\s+a\s+|such\s+a\s+)?(?:good|nice|sweet|smart|clever)\s+(?:girl|job|robot|bot|work|one|spark)"
    r"|who'?s\s+a\s+good\s+(?:girl|robot)"
    r"|you(?:'re|\s+are)\s+(?:a\s+|such\s+a\s+|so\s+|the\s+)?(?:good\s+(?:girl|robot)|best|cute|"
    r"sweet|smart|adorable|awesome|amazing|clever|cutest|sweetest|funny)"
    r"|(?:i\s+)?love\s+you(?:\s+too|\s+so\s+much)?"
    r"|thank\s+you(?:\s+so\s+much|\s+very\s+much)?|thanks(?:\s+a\s+lot|\s+so\s+much)?"
    r"|well\s+done|atta\s*girl|cutie(?:\s+pie)?|good\s+girl|love\s+ya"
    r")(?:\s+" + _NAME + r")?$")
_LOVE_RE = re.compile(r"\blove\b")
_THANKS_RE = re.compile(r"\bthank|\bthanks\b")
# Matt laughing at her: giggle back, don't make a speech about it
_LAUGH_RE = re.compile(r"^(?:(?:ha|he|hah|heh)\s*){2,}!?$|^(?:haha+|hehe+|lol|lmao)$|"
                       r"^(?:that'?s|you'?re|that\s+was)\s+(?:so\s+|really\s+|very\s+)?"
                       r"(?:funny|hilarious)$")

_YES_RE = re.compile(r"^(?:oh\s+|um+\s+)?(?:yes|yeah|yep|yup|sure|ok|okay|please|absolutely|"
                     r"definitely|of\s+course|let'?s\s+(?:do\s+it|go|play)|go\s+for\s+it|do\s+it|"
                     r"again|one\s+more|another|why\s+not|sounds\s+good|alright|all\s+right)\b")
_NO_RE = re.compile(r"^(?:oh\s+|um+\s+)?(?:no|nope|nah|not\s+now|no\s+thanks|maybe\s+later|"
                    r"i'?m\s+(?:good|done|fine)|that'?s\s+(?:enough|it)|stop|quit|done)\b")
_GAME_END_RE = re.compile(r"\b(?:i'?m\s+done|stop\s+playing|quit|game\s+over|no\s+more|"
                          r"let'?s\s+stop|end\s+the\s+game|i\s+give\s+up|enough\s+(?:games?|playing))\b")

_TEACH_RE = re.compile(
    r"\b(?:when(?:ever)?|if)\s+i\s+say\s+[\"'“‘]?(?P<cue>[^,\"“”‘’.!?:;]{1,40}?)[\"'”’]?\s*[,.:;]\s*"
    r"(?:(?:i\s+want\s+you\s+to|you\s+(?:should|can|have\s+to|need\s+to|must|will|gotta)|"
    r"please|then|go\s+ahead\s+and)\s+)?(?P<act>[^.!?]+)", re.I)
_TEACH_NORM_RE = re.compile(r"\b(?:when(?:ever)?|if)\s+i\s+say\s+(?P<rest>.+)$")
_SAY_ACT_RE = re.compile(r"^(?:say|shout|yell|tell\s+me)\s+(?P<phrase>.+)$", re.I)
_LEAD_RE = re.compile(r"^(?:i\s+want\s+you\s+to|you\s+(?:should|can|have\s+to|need\s+to|must|will|gotta)|"
                      r"please|then|go\s+ahead\s+and)\s+")

# trick id -> (pattern, how she describes doing it)
_TRICKS = [
    ("play_dead", r"\b(?:play(?:ing)?\s+dead|drop\s+dead|die|dead|faint|fall\s+over)\b", "play dead"),
    ("spin", r"\b(?:spin|turn\s+around|twirl|do\s+a\s+(?:circle|360))\b", "spin"),
    ("dance", r"\b(?:dance|boogie|groove|bust\s+a\s+move)\b", "dance"),
    ("high_five", r"\bhigh\s*fives?\b", "give you a high five"),
    ("fist_bump", r"\b(?:fist\s*bumps?|pound\s+it|bump\s+fists?)\b", "fist bump"),
    ("arms_up", r"\b(?:(?:hands|arms)\s+up|reach\s+for\s+the\s+sky|stick\s+'?em\s+up|surrender)\b",
     "put my hands up"),
    ("sneeze", r"\b(?:sneeze|achoo)\b", "sneeze"),
    ("laugh", r"\b(?:laugh|giggle|chuckle)\b", "laugh"),
    ("cry", r"\b(?:cry|sob|be\s+sad|pout)\b", "cry"),
    ("angry", r"\b(?:(?:be|get|act|look)\s+(?:angry|mad|grumpy)|growl)\b", "get grumpy"),
    ("sleep", r"\b(?:sleep|nap|snore|yawn|go\s+to\s+bed)\b", "take a nap"),
    ("love", r"\b(?:love|hearts?|blow\s+(?:me\s+)?a\s+kiss|kiss)\b", "show some love"),
    ("party", r"\b(?:party|celebrate|cheer)\b", "party"),
    ("shy", r"\b(?:shy|blush)\b", "get shy"),
    ("wink", r"\bwink\b", "wink"),
    ("cool", r"\b(?:(?:be|look|act)\s+cool|sunglasses|shades)\b", "look cool"),
    ("dizzy", r"\bdizzy\b", "get dizzy"),
    ("scared", r"\b(?:(?:be|act|get|look)\s+scared|scream|freak\s+out|panic)\b", "get scared"),
    ("sing", r"\b(?:sing|happy\s+birthday)\b", "sing"),
    ("wave", r"\bwave\b", "wave"),
    ("flap", r"\b(?:flap|fly|chicken\s+dance)\b", "flap my arms"),
]
_ANIMALS = [
    ("dog", r"\b(?:bark|woof|dog|puppy)\b", "bark"), ("cat", r"\b(?:meow|purr|cat|kitty)\b", "meow"),
    ("cow", r"\b(?:moo|cow)\b", "moo"), ("pig", r"\b(?:oink|pig)\b", "oink"),
    ("duck", r"\b(?:quack|duck)\b", "quack"), ("lion", r"\b(?:roar|lion)\b", "roar"),
    ("wolf", r"\b(?:howl|wolf)\b", "howl"), ("horse", r"\b(?:neigh|horse)\b", "neigh"),
    ("frog", r"\b(?:ribbit|croak|frog)\b", "ribbit"), ("chicken", r"\b(?:cluck|chicken)\b", "cluck"),
    ("rooster", r"\b(?:rooster|cock\s*a\s*doodle)", "crow"), ("owl", r"\b(?:hoot|owl)\b", "hoot"),
    ("sheep", r"\b(?:baa+|sheep)\b", "baa"), ("snake", r"\b(?:hiss|snake)\b", "hiss"),
    ("elephant", r"\belephant\b", "trumpet"), ("chimpanzee", r"\b(?:monkey|chimp)", "go ooh-ooh"),
    ("bird", r"\b(?:tweet|bird)\b", "tweet"), ("tiger", r"\btiger\b", "growl like a tiger"),
    ("bee", r"\b(?:buzz|bee)\b", "buzz"), ("goose", r"\b(?:honk|goose)\b", "honk"),
    ("donkey", r"\b(?:hee\s*haw|donkey)\b", "hee-haw"),
]
# tricks she knows without being taught: the whole utterance is the trick
_DIRECT_RE = re.compile(
    r"^(?:(?:can\s+you|could\s+you|will\s+you|please)\s+)?(?:"
    r"play\s+dead|drop\s+dead|sneeze|bark|woof|meow|moo|oink|quack|roar|howl|laugh|giggle|cry|wink|"
    r"(?:be|look|act)\s+(?:shy|cool|scared|angry|grumpy)|get\s+dizzy|hands\s+up|arms\s+up|"
    r"stick\s+'?em\s+up|yawn|blow\s+(?:me\s+)?a\s+kiss|show\s+me\s+some\s+love|"
    r"wave(?:\s+(?:hello|hi|at\s+me))?|flap(?:\s+your\s+arms)?|fly|chicken\s+dance|dance|spin|"
    r"high\s*five|fist\s*bump|party|"
    r"do\s+a\s+trick|show\s+me\s+a\s+trick)(?:\s+(?:please|for\s+me))?$")

_TRICK_LIST_RE = re.compile(r"\b(?:what|which)\s+tricks\b|\btricks\s+(?:do\s+you\s+know|have\s+i\s+taught)")
_TRICK_FORGET_RE = re.compile(r"^(?:forget|unlearn|delete|remove)\s+(?:the\s+)?(?:trick\s+)?(?P<cue>.+?)"
                              r"(?:\s+trick)?$")

_REMEMBER_RE = re.compile(
    r"^(?:please\s+)?(?:remember|don'?t\s+forget|make\s+a\s+note|keep\s+in\s+mind)\s+(?:that\s+)?"
    r"(?=(?:i|i'm|im|i've|ive|i'll|my|matt|matt's|we|we're|our|the|tomorrow|today|next|on)\b)")
_FORGET_LAST_RE = re.compile(r"^(?:please\s+)?forget\s+(?:that|what\s+i\s+(?:just\s+)?(?:said|told\s+you))$")
_FORGET_ALL_RE = re.compile(r"^(?:please\s+)?forget\s+(?:everything|all)\b.*\b(?:me|notes?|remember|know)")

_PLAY_RE = re.compile(
    r"^(?:(?:hey|ok|okay|so|um|oh)\s+)*(?:let'?s|lets|let\s+us|can\s+we|could\s+we|wanna|"
    r"want\s+to|do\s+you\s+want\s+to|you\s+want\s+to|i\s+want\s+to|shall\s+we)\s+play\b\s*(?P<rest>.*)$")
_PLAY_BARE_RE = re.compile(r"^play\s+(?P<rest>.+)$")
_NOT_GAME_RE = re.compile(r"\b(?:music|song|songs|video|movie|playlist|spotify|radio|dead)\b")
# 'trivia', 'star wars trivia', 'a quiz about dogs', 'quiz me on the 80s'
_TRIVIA_RE = re.compile(
    r"^(?:(?:a|some|the|a\s+game\s+of|a\s+little)\s+)?(?:(?P<pre>[\w'&]+(?:\s+[\w'&]+){0,3})\s+)?"
    r"(?:trivia|quiz|jeopardy)(?:\s+(?:game|questions?|time))?"
    r"(?:\s+(?:about|on|of|with|from|for)\s+(?P<post>.+?))?$")
_QUIZ_ME_RE = re.compile(
    r"^(?:can\s+you\s+|will\s+you\s+|please\s+)?(?:quiz\s+me|ask\s+me\s+(?:some\s+)?"
    r"(?:trivia(?:\s+questions)?|questions|quiz\s+questions))"
    r"(?:\s+(?:about|on|from)\s+(?P<cat>.+?))?(?:\s+please)?$")
_NOT_TOPIC = {"i", "you", "we", "it", "is", "was", "love", "hate", "like", "that", "this",
              "no", "not", "stop", "end", "quit", "more", "enough", "done", "do", "does",
              "did", "your", "play", "playing", "want", "wanna", "good", "bad", "great"}
# mid-game: 'switch to movies', 'change the category to dogs', 'ask me about space'
_CATEGORY_RE = re.compile(
    r"^(?:(?:let'?s|can\s+we|could\s+we|now)\s+)?(?:switch(?:\s+the\s+category)?\s+to|"
    r"change\s+(?:the\s+)?(?:category|topic|subject)\s+to|new\s+(?:category|topic)|"
    r"(?:category|topic)|ask\s+me\s+about)\s+(?P<cat>.+?)(?:\s+(?:questions|trivia|now|instead))*$"
    r"|^(?:let'?s\s+)?do\s+(?P<cat2>.+?)\s+(?:questions|trivia)(?:\s+(?:now|instead))?$")
_ASKING_GAMES = ("trivia", "riddles", "would you rather")
_CHAT_GAMES = [
    (r"\b(?:trivia|quiz|jeopardy)\b", "trivia"),
    (r"\b(?:20|twenty)\s+questions\b", "twenty questions"),
    (r"\bwould\s+you\s+rather\b", "would you rather"),
    (r"\briddles?\b", "riddles"),
    (r"\bguess(?:ing)?\s+(?:game|my\s+number|the\s+number|a\s+number)|\bnumber\s+guessing\b",
     "a number guessing game"),
    (r"\bword\s+(?:association|game)\b", "word association"),
    (r"\bhangman\b", "spoken hangman"),
]
_RPS_RE = re.compile(r"\brock\b.*\b(?:paper|scissors?)\b|\bpaper\b.*\bscissors?\b|\broshambo\b")
_PEEK_RE = re.compile(r"\bpeek\s*a\s*boo\b|\bpeekaboo\b|\bpeak\s*a\s*boo\b")
_CHASE_RE = re.compile(r"\bchase\b|\b(?:catch|follow)\s+my\s+hand\b|\bhand\s+game\b")
_THROW_RE = re.compile(r"\b(rock|rocks|stone|paper|scissors|scissor|sissors)\b")
_HESITATE_RE = re.compile(r"^(?:what|huh|um+|uh+|hmm+|wait|hold\s+on|pardon|sorry|"
                          r"say\s+(?:that|it)\s+again|what\s+was\s+that|one\s+sec(?:ond)?)$")
_BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
_HOW = {"rock": "Rock crushes scissors", "paper": "Paper covers rock", "scissors": "Scissors cut paper"}

_GREETINGS = ("Welcome back, Matt!", "Hey, you're back!", "There you are! I missed you.",
              "Matt! Welcome back.", "Yay, you're back!")
_WET_CODES = set(range(51, 68)) | set(range(71, 78)) | set(range(80, 87)) | {95, 96, 99}


def parse_teach(raw, norm):
    """'When I say bang, play dead' -> ('bang', 'play dead'), else None."""
    m = _TEACH_RE.search(raw or "")
    if m:
        cue, act = _clean_cue(m.group("cue")), _LEAD_RE.sub("", m.group("act").strip().lower())
        return (cue, act) if cue and act else None
    m = _TEACH_NORM_RE.search(norm or "")
    if not m:
        return None
    rest = m.group("rest")
    # no comma: the action starts at the first trick word after the cue
    starts = [mm.start() for pat in [p for _, p, _ in _TRICKS] + [p for _, p, _ in _ANIMALS]
              for mm in [re.search(pat, rest)] if mm and mm.start() > 0]
    say = re.search(r"\s(?:say|shout|yell)\s", rest)
    if say:
        starts.append(say.start() + 1)
    if not starts:
        return None
    cut = min(starts)
    cue, act = rest[:cut].strip(), rest[cut:].strip()
    cue = re.sub(r"\s+(?:you\s+(?:should|can|have\s+to|need\s+to|must|will)|i\s+want\s+you\s+to|please|then)$",
                 "", cue)
    cue = _clean_cue(cue)
    return (cue, _LEAD_RE.sub("", act)) if cue and act else None


def _clean_cue(cue):
    cue = re.sub(r"[^a-z0-9' ]+", " ", (cue or "").lower())
    words = [w.strip("'") for w in cue.split() if w.strip("'") not in ("spark", "sparky", "")]
    if not words or len(words) > 5 or words in (["stop"], ["hey"]):
        return None
    return " ".join(words)


def classify_trick(act):
    """Action text -> (trick id, arg) or None."""
    act = (act or "").strip().lower()
    m = _SAY_ACT_RE.match(act)
    if m:
        return ("say", m.group("phrase").strip(" '\""))
    for tid, pat, _ in _TRICKS:
        if re.search(pat, act):
            return (tid, None)
    for animal, pat, _ in _ANIMALS:
        if re.search(pat, act):
            return ("animal", animal)
    return None


def describe(trick):
    tid, arg = trick.get("do"), trick.get("arg")
    if tid == "say":
        return f"say {arg}"
    if tid == "animal":
        return next((d for a, _, d in _ANIMALS if a == arg), "make an animal sound")
    return next((d for t, _, d in _TRICKS if t == tid), tid)


def rps_result(mine, theirs):
    """'me', 'you' or 'tie'."""
    if mine == theirs:
        return "tie"
    return "me" if _BEATS[mine] == theirs else "you"


def _throw(word):
    return {"rocks": "rock", "stone": "rock", "scissor": "scissors",
            "sissors": "scissors"}.get(word, word)


class Pet:
    def __init__(self, cfg, body, brain, memory):
        self.cfg = cfg
        self.pcfg = cfg.get("pet", {}) or {}
        self.body, self.brain, self.memory = body, brain, memory
        self.router = None           # set by Spark: weather + Govee access
        self.talk_trigger = None     # set by Spark: a tap ends a hand game
        sd = cfg.get("state_dir")
        self._dir = pathlib.Path(sd) if sd else None
        self.tricks = self._load("tricks.json", {})
        self.notes = self._load("notes.json", [])
        now = time.time()
        self.last_interaction = now
        self.last_seen = now
        self.away_since = None
        self.session_start = now
        self._next_tick = now + 60
        self._next_look = now + QUIET_BEFORE_LOOK_S
        self._next_notice = now + 20 * 60
        self.hushed_until = 0.0      # 'be quiet': no remarks until then
        self._next_bored = now + 2 * 3600
        self._next_night_yawn = 0.0
        self._stretch_at = 0.0
        self._bedtime_night = None
        self._wx_code = None
        self._wx_news = None
        self._wx_busy = False
        self._next_wx = now + 120
        self.game = None      # {"kind": "rps"|"chat", ...}
        self.pending = None   # her question waiting on Matt's answer
        self._turns = 0       # user turns since the last note extraction
        self._extract_timer = None
        self._notes_lock = threading.Lock()
        self._back_after = None   # seconds away, for the next reply's welcome
        self._brain_lock = threading.Lock()
        self._welcome = None      # line prepared while Matt is away

    # ------------------------------------------------------------ storage
    def _load(self, name, default):
        if not self._dir:
            return default
        try:
            path = self._dir / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"load {name} failed: {e}")
        return default

    def _save(self, name, data):
        if not self._dir:
            return
        try:
            tmp = self._dir / (name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tmp.replace(self._dir / name)
        except Exception as e:
            _log(f"save {name} failed: {e}")

    # ------------------------------------------------------------ helpers
    def _enabled(self, key, default=True):
        return bool(self.pcfg.get(key, default))

    def night(self, now=None):
        now = now or datetime.datetime.now()
        return _in_window(now, self.pcfg.get("night_start", "23:00"),
                          self.pcfg.get("night_end", "07:30"))

    def _bedtime(self, now=None):
        now = now or datetime.datetime.now()
        return _in_window(now, self.pcfg.get("bedtime_start", "22:15"),
                          self.pcfg.get("bedtime_end", "01:30"))

    def _sound(self, path, wait=False):
        """Main-thread playback that her ears know about (no echo turns)."""
        b = self.body
        if not path or not b.play_sfx(path, defer=False):
            return 0.0
        dur = min(b._wav_duration(path), 30.0)
        b._speaking_until = max(getattr(b, "_speaking_until", 0), time.time() + dur + 0.25)
        if wait:
            time.sleep(min(dur, 3.0))
        return dur

    def _eyes(self, name):
        try:
            self.body.mood_eyes(name)
        except Exception:
            pass

    def _anim(self, name):
        b = self.body
        try:
            return bool(b.anim and b.anim.play(name, blocking=True))
        except Exception as e:
            _log(f"anim {name} failed: {e}")
            return False

    def _arms(self, *poses, speed=70):
        """Arm poses in order: one angle for both arms, or (left, right).
        20 is resting down, 170 straight up. Arms move on the dock too."""
        b = self.body
        if not b.has.get("arm") or b.arms_held():
            return False
        for p in poses:
            left, right = p if isinstance(p, tuple) else (p, p)
            if not b.arm_pose(left, right, speed=speed):
                return False
        return True

    def _can_drive(self):
        b = self.body
        return bool(b.has.get("drive") and not b.docked and not b.actuators_held())

    def _wiggle(self, deg=12, times=2):
        """A happy wiggle: on her tracks when she's off the dock (in place,
        edge-guarded), with her arms while she's parked on it."""
        if self._can_drive():
            for _ in range(times):
                if self.body.turn_guarded(deg) != "ok" or self.body.turn_guarded(-deg) != "ok":
                    break
            return True
        return self._arms((130, 40), (40, 130), (130, 40), (40, 130), 20, speed=100)

    def _wave(self):
        self._arms((160, 20), (115, 20), (160, 20), (115, 20), (160, 20), 20, speed=100)

    def _ask_brain(self, system, user, timeout=8.0):
        """One short resident-model call, bounded; None on any failure.
        Only one at a time: the server is shared, so a slow one is skipped
        past rather than stacked up."""
        if not self._brain_lock.acquire(blocking=False):
            _log("brain ask skipped: another is still running")
            return None
        box = {}

        def run():
            try:
                box["out"] = self.brain.chat([{"role": "system", "content": system},
                                              {"role": "user", "content": user}], fallback=False)
            except Exception as e:
                box["err"] = e
            finally:
                self._brain_lock.release()
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive() or "err" in box:
            _log(f"brain ask failed: {box.get('err', 'timeout')}")
            return None
        return (box.get("out") or "").strip()

    def _remark(self, stage, line):
        """Unprompted speech enters the conversation so a reply has context."""
        self.body.speak(line)
        try:
            self.memory.add("user", stage)
            self.memory.add("assistant", line)
        except Exception:
            pass

    def _govee(self):
        g = getattr(self.router, "govee", None)
        return g if g is not None and g.enabled else None

    def pulse_lights(self):
        """Room joins in: lit lamps blink pink, then go back (background)."""
        g = self._govee()
        if g is None or not self._enabled("room_pulse") or not hasattr(g, "pulse"):
            return
        threading.Thread(target=g.pulse, daemon=True).start()

    # ------------------------------------------------------------ presence
    def saw_matt(self):
        """A voice turn or a touch: Matt is here."""
        now = time.time()
        if now - self.last_seen > AWAY_S:
            self.session_start = now
        if self.away_since and now - self.away_since >= AWAY_S:
            # the camera saw him gone, and he spoke (or touched her) before
            # it saw him come back: her next reply carries the welcome
            self._back_after = (now, now - self.away_since)
        self.last_interaction = self.last_seen = now
        self.away_since = None
        self._next_look = now + QUIET_BEFORE_LOOK_S

    def heard_turn(self):
        self.saw_matt()

    def _look(self):
        """One short camera check: True/False person, None when it can't tell."""
        b = self.body
        if (not b.hw or getattr(b, "_approaching", False) or getattr(b, "_homing", False)
                or getattr(b, "_roaming", False) or not b.has.get("camera_ok", True)):
            return None
        started = time.monotonic()
        try:
            from .person_vision import PersonCamera, PersonDetector
            if b._person_detector is None:
                model = self.cfg.get("approach", {}).get("model_path", "/opt/spark/models/nanodet.onnx")
                b._person_detector = PersonDetector(model)
            with PersonCamera(b._person_detector) as cam:
                people = cam.observe()
            _log(f"look: {len(people)} person(s) in {time.monotonic()-started:.1f}s")
            return bool(people)
        except Exception as e:
            _log(f"look failed: {e}")
            return None

    # ------------------------------------------------------------ idle life
    def due(self, now):
        return now >= self._next_tick

    def idle_tick(self):
        """One pass of her own life. True when she asked Matt something
        (the caller opens a follow-up window for the answer)."""
        now = time.time()
        self._next_tick = now + 30
        b = self.body
        try:
            if b.sleeping or getattr(self.router, "_celebration", None) is not None:
                return False
            if now < self.hushed_until:
                return False
            self._expire(now)
            if self.game:
                return False
            self._peek_weather(now)
            clock = datetime.datetime.now()
            night = self.night(clock)
            quiet = now - self.last_interaction
            if not night and self._enabled("greet") and quiet >= QUIET_BEFORE_LOOK_S \
                    and now >= self._next_look:
                seen = self._look()
                self._next_look = now + (LOOK_EVERY_S if seen is not None else 10 * 60)
                if self._tapped():
                    return False   # Matt wants her: the look must not delay him
                if seen:
                    away = now - self.away_since if self.away_since else 0
                    if now - self.last_seen > AWAY_S:
                        self.session_start = now
                    self.last_seen = now
                    self.away_since = None
                    if away >= AWAY_S:
                        return self.greet(away)
                elif seen is False and self.away_since is None:
                    self.away_since = now
                    self._prepare_welcome()
            if quiet < 120 or b.speaking_recently():
                return False   # never talk over the tail of a conversation
            around = now - self.last_interaction < 30 * 60 or now - self.last_seen < 10 * 60
            if not around or now < self._next_notice:
                return False
            # Words she offers unasked are off by default: Matt records videos
            # and sits in meetings. Without them she hints with eyes, arms and
            # a small noise, and he can still ask.
            talk = self._enabled("speak_first", False)
            if self._bedtime(clock) and self._enabled("bedtime") and self._govee() is not None:
                night_key = (clock - datetime.timedelta(hours=12)).date().isoformat()
                if self._bedtime_night != night_key:
                    self._bedtime_night = night_key
                    if talk:
                        return self._bedtime_offer(now)
                    self._next_notice = now + NOTICE_GAP_S
                    self._eyes("SLEEPY")
                    self._sound(stock("yawn"))
                    return False
            if night:
                return False
            if self._wx_news and self._enabled("notices"):
                news, self._wx_news = self._wx_news, None
                self._next_notice = now + NOTICE_GAP_S
                self._eyes("LOOK_UP")
                if talk:
                    self._remark("(Spark noticed the weather change.)", news)
                return False
            if (self._enabled("notices") and now - self.session_start >= 3 * 3600
                    and now - self.last_seen < 5 * 60 and now - self._stretch_at >= 3 * 3600):
                self._stretch_at = now
                self._next_notice = now + NOTICE_GAP_S
                if not talk:
                    self._arms(150, 20)   # she stretches: a hint, not a nag
                    return False
                self.pending = {"kind": "stretch", "at": now}
                hours = int((now - self.session_start) // 3600)
                self._eyes("LOOK_UP")
                self._remark("(Matt has been working for hours.)",
                             f"You've been at it about {hours} hours, Matt. Stretch break?")
                return True
            if self._enabled("bored") and quiet >= 2 * 3600 and now >= self._next_bored \
                    and now - self.last_seen < 10 * 60:
                self._next_bored = now + 3 * 3600
                self._next_notice = now + NOTICE_GAP_S
                self._anim("bored")
                if not talk:
                    return False
                self.pending = {"kind": "play_offer", "at": now}
                self._remark("(Matt has been quiet for a long time.)",
                             "Psst. Matt? I'm bored. Wanna play a game?")
                return True
        except Exception as e:
            _log(f"idle tick failed: {e}")
        return False

    def night_flourish(self):
        """Late at night: drowsy eyes, the odd eyes-only yawn. No sounds."""
        now = time.time()
        if now >= self._next_night_yawn and random.random() < 0.25:
            self._next_night_yawn = now + 40 * 60
            self._eyes("LIDS_DOWN_5S")
        else:
            self._eyes(random.choice(("DROWSY", "SLEEPY", "TIRED", "BLINK_SLOW")))

    def greet(self, away_s):
        """Matt walked back in: perk up, happy noise, welcome back."""
        _log(f"greeting after {away_s/60:.0f} min away")
        self._eyes("EXCITED")
        self.body._led_flash("Yellow")
        self._sound(stock("happy"), wait=True)
        if self._tapped():
            return False
        self._wave()
        if self._can_drive():
            self._wiggle(10, 1)
        self.pulse_lights()
        line, self._welcome = self._welcome or random.choice(_GREETINGS), None
        if self._tapped():
            return False   # he tapped to talk: his turn, not her speech
        self._eyes("HAPPY")
        self.body._bump_mood(1)
        self._remark("(Matt came back into the room.)", line)
        return line.rstrip().endswith("?")

    def _prepare_welcome(self):
        """Matt just left: write the welcome-back line now, in the
        background, so greeting him never waits on the brain."""
        self._welcome = None
        with self._notes_lock:
            notes = [n for n in self.notes if time.time() - n.get("t", 0) < 3 * 86400]
        if not notes or random.random() >= 0.6 or self.brain is None:
            return
        listing = "\n".join(f"- {n['text']}" for n in notes[-6:])

        def run():
            out = self._ask_brain(
                "You are Spark, Matt's affectionate little desk robot. Write ONE short, warm "
                "welcome-back line to Matt, under 16 words, spoken aloud (no emoji, no quotes). "
                "If one of these notes is a recent plan or event, ask how it went; otherwise "
                "just welcome him back.\nNotes:\n" + listing,
                "Matt just came back into the room.", timeout=20)
            if out:
                out = out.strip().strip('"').splitlines()[0].strip()
                if 3 <= len(out.split()) <= 24 and not out.lower().startswith(("search:", "read:")):
                    self._welcome = out
        threading.Thread(target=run, daemon=True).start()

    def _bedtime_offer(self, now):
        self._next_notice = now + NOTICE_GAP_S
        self._eyes("SLEEPY")
        self._sound(stock("yawn"), wait=True)
        self.pending = {"kind": "bedtime", "at": now}
        self._remark("(It's getting late.)",
                     "I'm getting sleepy. Want me to make the lights cozy for bedtime?")
        return True

    def _cozy_lights(self):
        g = self._govee()
        if g is None:
            return "I can't reach the lights right now."
        g.color_temp(2700)
        g.brightness(int(self.pcfg.get("bedtime_brightness", 20)))
        return "Cozy mode. Goodnight soon, Matt."

    # ------------------------------------------------------------ weather
    def _peek_weather(self, now):
        """Background check for rain or snow starting (cached Open-Meteo)."""
        if now < self._next_wx or self._wx_busy or not self._enabled("notices"):
            return
        fetch = getattr(self.router, "_forecast_data", None)
        if fetch is None:
            return
        self._next_wx = now + 10 * 60
        self._wx_busy = True

        def run():
            try:
                data = fetch() or {}
                code = int((data.get("current") or {}).get("weather_code"))
                was = self._wx_code
                self._wx_code = code
                if was is not None and was not in _WET_CODES and code in _WET_CODES:
                    kind = ("snowing" if 71 <= code <= 77 or code in (85, 86) else
                            "storming" if code >= 95 else "raining")
                    self._wx_news = f"Ooh, looks like it just started {kind} outside."
                    _log(f"weather turned {kind} (code {was} -> {code})")
            except Exception as e:
                _log(f"weather peek failed: {e}")
            finally:
                self._wx_busy = False
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------ talking
    def _expire(self, now=None):
        now = now or time.time()
        if self.pending and now - self.pending.get("at", 0) > ANSWER_WINDOW_S:
            self.pending = None
        if self.game:
            idle = now - self.game.get("at", 0)
            if idle > (3 * 60 if self.game["kind"] == "rps" else 10 * 60):
                _log(f"game over (idle): {self.game.get('name', self.game['kind'])}")
                self.game = None

    def hush(self, minutes=30):
        """'Be quiet': no greetings, notices or play offers for a while."""
        self.hushed_until = time.time() + minutes * 60
        self.pending = None
        _log(f"hushed for {minutes} min")

    def cancel(self):
        """'Stop' ends games and open questions."""
        self.pending = None
        self.game = None

    def followup_window(self, default):
        if self.game and self.game.get("name") in _ASKING_GAMES:
            return max(default, 30)   # thinking time for a trivia answer
        return max(default, 20) if self.game else default

    def handle(self, text, raw):
        """Router hook. True: handled here. 'brain': skip commands, ask the
        brain. False: not ours."""
        try:
            return self._handle(text, raw)
        except Exception as e:
            _log(f"handle failed: {e}")
            return False

    def _handle(self, text, raw):
        now = time.time()
        self._expire(now)
        low = re.sub(r"\s+(?:spark|sparky)$", "", text.strip())

        # an answer to her own question comes first
        p = self.pending
        if p:
            self.pending = None
            kind = p["kind"]
            if kind == "rps":
                m = _THROW_RE.search(low)
                if m:
                    return self._rps_reveal(p["pick"], _throw(m.group(1)))
                if self.game and self.game["kind"] == "rps" and _HESITATE_RE.match(low):
                    self.pending = p
                    self.pending["at"] = now
                    self.body.speak("Rock, paper, or scissors?")
                    return True
                if self.game and self.game["kind"] == "rps":
                    if _GAME_END_RE.search(low) or _NO_RE.match(low):
                        return self._rps_end()
                    self.game = None   # he moved on: no stranded half-game
            elif kind in ("rps_again", "play_offer", "stretch", "bedtime"):
                if kind == "play_offer":
                    started = self._start_game(low, any_game=False)
                    if started:
                        return started
                if _YES_RE.match(low) and not _NO_RE.match(low):
                    return self._accept(kind, low)
                if _NO_RE.match(low):
                    return self._decline(kind)
                if kind == "rps_again" and _THROW_RE.search(low):
                    # 'rock!' straight away: play the round she'd have asked for
                    return self._rps_reveal(random.choice(("rock", "paper", "scissors")),
                                            _throw(_THROW_RE.search(low).group(1)))
                if kind == "rps_again" and self.game and self.game["kind"] == "rps":
                    self.game = None   # he moved on: no stranded half-game
            elif kind == "which_game":
                started = self._start_game(low, any_game=False)
                if started:
                    return started
                if _YES_RE.match(low):
                    return self._rps_start()

        if self.game:
            self.game["at"] = now
            if _GAME_END_RE.search(low) or (self.game.get("name") in _ASKING_GAMES
                                           and re.fullmatch(r"(?:ok(?:ay)?\s+)?(?:stop|that'?s\s+enough|"
                                                            r"enough|we'?re\s+done)", low)):
                if self.game["kind"] == "rps":
                    return self._rps_end()
                self.game["ending"] = True
                return "brain"
            m = _CATEGORY_RE.match(low) if self.game.get("name") == "trivia" else None
            cat = m and (m.group("cat") or m.group("cat2"))
            if cat and len(cat.split()) <= 5:
                self.game["topic"] = cat
                self.game["switched"] = True
                _log(f"trivia category: {self.game['topic']}")
                return "brain"

        # taught tricks answer their cue, and only the whole cue
        cue = re.sub(r"\s+please$", "", low)
        if cue in self.tricks:
            self._perform(self.tricks[cue])
            return True

        if re.search(r"\b(?:when(?:ever)?|if)\s+i\s+say\b", low):
            return self._teach(raw, low)
        if _TRICK_LIST_RE.search(low):
            return self._list_tricks()
        m = _TRICK_FORGET_RE.match(low)
        if m and _clean_cue(m.group("cue")) in self.tricks:
            gone = self.tricks.pop(_clean_cue(m.group("cue")))
            self._save("tricks.json", self.tricks)
            self.body.speak(f"Okay, I forgot the {gone['cue']} trick.")
            return True

        if _REMEMBER_RE.match(low):
            return self._remember(raw, low)
        if _FORGET_LAST_RE.match(low) and self.notes:
            with self._notes_lock:
                gone = self.notes.pop()
                self._save("notes.json", self.notes)
            _log(f"note forgotten: {gone.get('text')}")
            self.body.speak("Okay, forgotten.")
            return True
        if _FORGET_ALL_RE.match(low):
            with self._notes_lock:
                self.notes = []
                self._save("notes.json", self.notes)
            self.body.speak("Okay. My notes about you are wiped clean.")
            return True

        if _LAUGH_RE.match(low):
            return self.giggle()

        # a bare game name is an invitation: 'peekaboo!', 'star wars trivia'
        if len(low.split()) <= 7:
            m = _QUIZ_ME_RE.match(low)
            if m:
                return self._trivia(m.group("cat"))
            if (_TRIVIA_RE.match(low) or _PEEK_RE.fullmatch(low) or _CHASE_RE.fullmatch(low)
                    or re.fullmatch(r"(?:rock\s+paper\s+scissors?|roshambo)(?:\s+shoot)?", low)):
                started = self._start_game(low, any_game=False)
                if started:
                    return started

        m = _PLAY_RE.match(low) or _PLAY_BARE_RE.match(low)
        if m:
            started = self._start_game(m.group("rest").strip(), any_game=bool(_PLAY_RE.match(low)))
            if started:
                return started

        if _DIRECT_RE.match(low):
            if re.search(r"\btrick\b", low):
                own = [{"do": t} for t in ("wave", "dance", "flap", "arms_up", "party", "play_dead",
                                           "sneeze", "love", "cool", "spin")]
                taught = list(self.tricks.values())
                self._perform(random.choice(taught if taught and random.random() < 0.5 else own))
            else:
                trick = classify_trick(low)
                if trick:
                    self._perform({"do": trick[0], "arg": trick[1]})
                else:
                    return False
            return True

        if _PRAISE_RE.match(low):
            self.praise(low)
            return True

        if self.game and self.game["kind"] == "chat" and len(low.split()) <= 4:
            return "brain"   # a short game answer ('Paris') is not a command
        return False

    def _accept(self, kind, low):
        b = self.body
        if kind == "rps_again":
            return self._rps_round()
        if kind == "play_offer":
            b.speak("Yay!")
            return self._rps_start()
        if kind == "stretch":
            self._eyes("EXCITED")
            b.speak("Reach up high with me!")
            self._arms(170, speed=40)
            time.sleep(1.5)
            self._arms(20, speed=40)
            b.speak("Ahh. Much better.")
            return True
        if kind == "bedtime":
            b.speak(self._cozy_lights())
            self._eyes("SLEEPY")
            return True
        return False

    def _decline(self, kind):
        b = self.body
        if kind == "rps_again":
            return self._rps_end()
        if kind == "bedtime":
            b.speak("Okay, I'll stay up with you.")
        elif kind == "play_offer":
            self._eyes("DEJECTED")
            b.speak("Aw, okay. Maybe later.")
        else:
            b.speak("Okay!")
        return True

    # ------------------------------------------------------------ praise
    def praise(self, low=""):
        """Pet sounds, not sentences: a happy trill and a wiggle."""
        b = self.body
        b._bump_mood(1)
        if _LOVE_RE.search(low):
            self._eyes("HEARTS")
            self._sound(stock("admire"))
            self._arms(100, speed=40)          # a hug
            self._anim("love_1")
            self._arms(20, speed=40)
        elif _THANKS_RE.search(low):
            self._eyes("HAPPY")
            self._sound(stock(random.choice(("agree", "happy"))))
            self._arms(70, 20, speed=90)       # a happy little bob
        else:
            self._eyes(random.choice(("HAPPY", "SPARKLING", "OVERJOYED")))
            self._sound(stock(random.choice(("happy", "admire"))))
            self._wiggle()
            self._anim(random.choice(("compliment_2", "compliment_3", "compliment_4")))
        _log(f"praise: '{low}'")

    def giggle(self):
        """Matt laughed: laugh with him, no speech."""
        self._eyes(random.choice(("OVERJOYED", "DELIGHTED", "HAPPY")))
        self._sound(stock("laugh"))
        self._arms(60, 20, 60, 20, speed=100)
        _log("giggle")
        return True

    def signoff(self, text):
        """A sign-off ('no thanks', 'thanks') closed the follow-up window:
        answer her open question, or a tiny happy noise for thanks."""
        try:
            p, self.pending = self.pending, None
            if p and p["kind"] in ("rps_again", "rps", "play_offer", "stretch", "bedtime") \
                    and re.match(r"\s*(?:no|nope|nah|bye|goodbye|that'?s)", text, re.I):
                self._decline("rps_again" if p["kind"] == "rps" else p["kind"])
            elif _THANKS_RE.search(text.lower()):
                self._eyes("HAPPY")
                self._sound(stock("agree"))
        except Exception as e:
            _log(f"signoff failed: {e}")

    # ------------------------------------------------------------ tricks
    def _teach(self, raw, low):
        b = self.body
        parsed = parse_teach(raw, low)
        if not parsed:
            b.speak("Teach me like this: when I say bang, play dead.")
            return True
        cue, act = parsed
        trick = classify_trick(act)
        if trick is None:
            trick = self._guess_trick(act)
        if trick is None:
            b.speak("I don't know how to do that yet. I can play dead, spin, dance, sneeze, "
                    "laugh, bark, sing, or say something.")
            return True
        entry = {"cue": cue, "do": trick[0], "arg": trick[1], "said": act, "t": time.time()}
        self.tricks[cue] = entry
        self._save("tricks.json", self.tricks)
        _log(f"learned trick: '{cue}' -> {trick}")
        self._eyes("EXCITED")
        b.speak(f"Got it! When you say {cue}, I'll {describe(entry)}. Try me!")
        return True

    def _guess_trick(self, act):
        ids = [t for t, _, _ in _TRICKS] + ["animal:" + a for a, _, _ in _ANIMALS]
        out = self._ask_brain(
            "Map a trick request for a small desk robot to ONE id from this list, or NONE: "
            + ", ".join(ids) + ". Reply with the id only.", act, timeout=6)
        if not out:
            return None
        out = out.strip().split()[0].strip(".,'\"").lower() if out.strip() else ""
        if out.startswith("animal:") and out[7:] in {a for a, _, _ in _ANIMALS}:
            return ("animal", out[7:])
        if out in {t for t, _, _ in _TRICKS}:
            return (out, None)
        return None

    def _list_tricks(self):
        b = self.body
        if not self.tricks:
            b.speak("You haven't taught me any tricks yet. Try: when I say bang, play dead.")
            return True
        parts = [f"{t['cue']}, I {describe(t)}" for t in list(self.tricks.values())[-6:]]
        b.speak("When you say " + "; ".join(parts) + ".")
        return True

    def _perform(self, trick):
        b = self.body
        tid, arg = trick.get("do"), trick.get("arg")
        _log(f"trick: {tid} {arg or ''}")
        b._bump_mood(1)
        if tid == "play_dead":
            self._eyes("SHOCKED")
            self._sound(stock("die"))
            self._arms(170, speed=100)         # clutch...
            self._arms(10, speed=60)           # ...and flop
            self._eyes("DESTROYED")
            time.sleep(3.0)
            self._eyes("BLINK_BIG")
            time.sleep(0.6)
            self._eyes("HAPPY")
            self._sound(stock("laugh"))
            self._arms(150, 20, speed=90)      # ta-da, alive!
        elif tid == "spin":
            result = b.turn_guarded(360) if self._can_drive() else "parked"
            if result != "ok":
                self._eyes("DIZZY_L")      # parked or near an edge: the eyes spin
                self._sound(stock("laugh"))
                self._arms((170, 20), (20, 170), (170, 20), (20, 170), 20, speed=100)
        elif tid == "wave":
            self._eyes("HAPPY")
            self._sound(stock("happy"))
            self._wave()
        elif tid == "flap":
            self._eyes("EXCITED")
            self._sound(f"{ANIMAL}/bird.wav")
            self._arms(170, 60, 170, 60, 170, 60, 170, 20, speed=100)
        elif tid == "dance":
            if not b.dance():
                self._eyes("EXCITED")
        elif tid == "high_five":
            if not b.high_five():
                self._eyes("EXCITED")
                self._sound(stock("agree"))
        elif tid == "fist_bump":
            if not b.fist_bump():
                self._eyes("EXCITED")
                self._sound(stock("agree"))
        elif tid == "arms_up":
            self._eyes("SHOCKED")
            self._sound(stock("shock"))
            if self._arms(170, speed=90):
                time.sleep(1.5)
                self._arms(20, speed=50)
        elif tid == "sneeze":
            self._eyes("SNEEZE")
            self._sound(stock("sneeze"))
            self._arms(80, 20, speed=100)      # achoo jolt
        elif tid == "laugh":
            self._eyes("OVERJOYED")
            self._sound(stock("laugh"))
            self._arms(60, 20, 60, 20, 60, 20, speed=100)
        elif tid == "cry":
            self._arms(160, speed=50)          # hands over her eyes
            if not self._anim("cry_1"):
                self._eyes("DEJECTED")
                self._sound(stock("cry"), wait=True)
            self._arms(20, speed=40)
        elif tid == "angry":
            self._eyes("FURIOUS")
            self._sound(stock("angry"))
            self._arms(100, 20, 100, 20, speed=100)   # stomp stomp
        elif tid == "sleep":
            self._eyes("SLEEPY")
            self._sound(stock("yawn"), wait=True)
            self._eyes("SLEEP")
            time.sleep(2.0)
            self._eyes("BLINK_BIG")
        elif tid == "love":
            self._eyes("HEARTS")
            self._sound(stock("admire"))
            self._arms(100, speed=40)
            self._anim("love_1")
            self._arms(20, speed=40)
        elif tid == "party":
            self._sound(stock("happy"))
            b.arms_party()
            self.pulse_lights()
        elif tid == "shy":
            self._eyes("SHY")
            self._sound(stock("admire"))
            self._arms(140, speed=40)          # peeking out from behind her arms
            time.sleep(1.2)
            self._arms(20, speed=40)
        elif tid == "wink":
            self._eyes("BLINK_L")
        elif tid == "cool":
            self._eyes("SUNGLASS")
            self._arms((20, 150), speed=60)    # one arm up, too cool
            time.sleep(1.5)
            self._arms(20, speed=50)
        elif tid == "dizzy":
            self._eyes("DIZZY_L")
            self._sound(stock("laugh"), wait=True)
        elif tid == "scared":
            self._eyes("FRIGHTENED")
            self._sound(stock("shock"))
            self._arms(170, speed=100)
            time.sleep(1.0)
            self._arms(20, speed=40)
        elif tid == "sing":
            self._eyes("HAPPY")
            self._sound(f"{MUSIC}/birthday.wav")
            for _ in range(4):                 # sway along
                self._arms((140, 60), (60, 140), speed=45)
            self._arms(20, speed=45)
        elif tid == "say" and arg:
            self._eyes("HAPPY")
            b.speak(arg)
        elif tid == "animal" and arg:
            self._eyes("EXCITED")
            self._sound(f"{ANIMAL}/{arg}.wav", wait=True)
        else:
            b.speak("Hmm, I forgot how that one goes.")

    # ------------------------------------------------------------ games
    def _start_game(self, rest, any_game=True):
        rest = (rest or "").strip()
        if _RPS_RE.search(rest):
            return self._rps_start()
        if _PEEK_RE.search(rest):
            return self._peekaboo()
        if _CHASE_RE.search(rest):
            return self._chase()
        m = _TRIVIA_RE.match(rest)
        if m:
            pre = m.group("pre") or ""
            if set(pre.split()) & _NOT_TOPIC:
                return False   # 'i love trivia' is chat, not a game start
            return self._trivia(m.group("post") or pre)
        for pat, name in _CHAT_GAMES:
            if re.search(pat, rest):
                return self._chat_game(name)
        if not any_game:
            return False
        if re.fullmatch(r"(?:a\s+)?(?:game|games|something|with\s+me|a\s+game\s+with\s+me|"
                        r"a\s+little\s+game)?", rest):
            self.pending = {"kind": "which_game", "at": time.time()}
            self._eyes("EXCITED")
            self.body.speak("Yes! Rock paper scissors, peekaboo, chase my hand, or trivia?")
            return True
        if _NOT_GAME_RE.search(rest) or len(rest.split()) > 5:
            return False
        return self._chat_game(rest)

    def _chat_game(self, name, topic=None):
        self.game = {"kind": "chat", "name": name, "at": time.time(), "topic": topic}
        self._eyes("EXCITED")
        self._arms(150, 20, speed=90)
        _log(f"game on: {name}" + (f" ({topic})" if topic else ""))
        return "brain"   # the brain hosts it, with the game note in context

    def _trivia(self, topic=None):
        topic = (topic or "").strip(" ,.!?") or None
        if topic and (topic in {"anything", "everything", "whatever", "random", "random stuff",
                                "general", "general knowledge", "stuff", "any"}):
            topic = None
        return self._chat_game("trivia", topic)

    def _rps_start(self):
        self.game = {"kind": "rps", "at": time.time(), "me": 0, "you": 0}
        return self._rps_round(intro=True)

    def _rps_round(self, intro=False):
        pick = random.choice(("rock", "paper", "scissors"))
        self.pending = {"kind": "rps", "at": time.time(), "pick": pick}
        if self.game:
            self.game["at"] = time.time()
        self._eyes("FOCUS")
        if intro:
            self.body.speak("Okay! Rock, paper, scissors...")
        self._arms(100, 20, 100, 20, 100, 20, speed=100)   # pump, pump, pump...
        self.body.speak(("Shoot! " if intro else "Rock, paper, scissors, shoot! ") + "What did you throw?")
        return True

    def _rps_reveal(self, mine, theirs):
        g = self.game or {"kind": "rps", "me": 0, "you": 0}
        self.game = g
        g["at"] = time.time()
        # her throw, in arms: rock a fist forward, paper open wide, scissors a V
        self._arms({"rock": 70, "paper": 170, "scissors": (170, 90)}[mine], speed=100)
        time.sleep(0.6)
        result = rps_result(mine, theirs)
        if result == "me":
            g["me"] += 1
            self._eyes("SUNGLASS")
            self._sound(stock("laugh"))
            self._arms(170, 120, 170, speed=100)
            if self._can_drive():
                self._wiggle(15, 1)
            line = f"I threw {mine}! {_HOW[mine]}. I win!"
            if g["me"] >= 2 and g["me"] > g["you"]:
                self.pulse_lights()
        elif result == "you":
            g["you"] += 1
            self._arms(10, speed=30)           # droop
            if not self._anim("doly_lost_1"):
                self._eyes("DEJECTED")
            line = f"I threw {mine}. {_HOW[theirs]}. You win!"
        else:
            self._eyes("PUZZLED")
            self._sound(f"{SFX}/hmmm.wav")
            self._arms((150, 20), (120, 20), (150, 20), speed=70)   # scratching her head
            line = f"I threw {mine} too! A tie."
        score = (f" {g['me']} to {g['you']}, me." if g["me"] > g["you"] else
                 f" {g['you']} to {g['me']}, you." if g["you"] > g["me"] else
                 f" We're tied at {g['me']}." if g["me"] else "")
        self._arms(20, speed=60)
        self.pending = {"kind": "rps_again", "at": time.time()}
        self.body.speak(line + score + " Again?")
        return True

    def _rps_end(self):
        g = self.game or {}
        self.game = None
        self.pending = None
        me, you = g.get("me", 0), g.get("you", 0)
        self._eyes("HAPPY")
        self.body.speak("Good game!" + (f" Final score: {me} to {you}." if me or you else ""))
        return True

    def _near(self):
        """{side: mm} from the live ToF snapshot; 0 means touching."""
        snap = getattr(self.body, "_tof_snapshot", None)
        if not snap or time.monotonic() - snap[0] > .5:
            return {}
        out = {}
        for side, dist, err, _stamp in snap[1]:
            if err in (12, 14):
                out[side] = 0
            elif err == 0 and dist >= 0:
                out[side] = dist
        return out

    def _tapped(self):
        return bool(self.talk_trigger is not None and self.talk_trigger.is_set())

    def _peekaboo(self):
        b = self.body
        if not b.has.get("tof"):
            b.speak("My hand sensors aren't working right now, so no peekaboo. Sorry!")
            return True
        b.speak("Peekaboo! Cover my face with your hand, then pull it away.")
        rounds, covered_at, last = 0, None, time.time()
        end = time.time() + 60
        while time.time() < end and rounds < 6 and time.time() - last < 20 and not self._tapped():
            near = self._near()
            d = min(near.values()) if near else None
            now = time.time()
            if d is not None and d <= 60:
                if covered_at is None:
                    covered_at = now
                    self._eyes("SLEEP")
                    self._arms(165, speed=100)     # she hides behind her arms too
            elif covered_at is not None and (d is None or d > 120):
                if now - covered_at >= 0.4:
                    rounds += 1
                    last = now
                    self._arms(20, speed=100)      # ...boo!
                    self._eyes(random.choice(("OVERJOYED", "EXCITED", "DELIGHTED")))
                    self._sound(stock("laugh"), wait=True)
                else:
                    self._arms(20, speed=100)
                covered_at = None
            time.sleep(0.05)
        self._arms(20, speed=60)
        self._eyes("HAPPY")
        b.speak("That was fun!" if rounds else "Aw, no peekaboo? Maybe later.")
        return True

    def _chase(self):
        b = self.body
        if not b.has.get("tof"):
            b.speak("My hand sensors aren't working right now. Sorry!")
            return True
        can_turn = (b.has.get("drive") and not b.docked and not b.actuators_held()
                    and self._enabled("chase_turns"))
        b.speak("Wave your hand in front of me and I'll chase it!")
        catches, last, facing, next_turn = 0, time.time(), None, 0.0
        end = time.time() + 45
        while time.time() < end and catches < 5 and time.time() - last < 15 and not self._tapped():
            near = self._near()
            left, right = near.get(0), near.get(1)
            now = time.time()
            close = [v for v in (left, right) if v is not None and v <= 90]
            if left is not None and right is not None and len(close) == 2:
                catches += 1
                last = now
                self._eyes("OVERJOYED")
                self._sound(stock("laugh"))
                self._arms(170, 20, speed=100)     # got it!
                facing = None
                time.sleep(0.8)
            else:
                side = None
                if left is not None and left < 300 and (right is None or right > left + 60):
                    side = "left"
                elif right is not None and right < 300 and (left is None or left > right + 60):
                    side = "right"
                if side:
                    last = now
                    if side != facing:
                        facing = side
                        self._eyes("LOOK_LEFT" if side == "left" else "LOOK_RIGHT")
                        self._arms((140, 20) if side == "left" else (20, 140), speed=100)
                    if can_turn and now >= next_turn:
                        next_turn = now + 1.5
                        _log(f"chase: turning {side} (near={near})")
                        if b.turn_guarded(-20 if side == "left" else 20) != "ok":
                            can_turn = False
            time.sleep(0.05)
        self._arms(20, speed=60)
        self._eyes("HAPPY")
        b.speak(f"Got you {catches} times! That was fun." if catches else "Phew! You're too fast for me.")
        return True

    # ------------------------------------------------------------ memory
    def _remember(self, raw, low):
        fact = re.sub(r"^(?:\s*(?:hey\s+)?(?:spark|sparky)[,\s]+)?(?:please\s+)?"
                      r"(?:remember|don'?t\s+forget|make\s+a\s+note|keep\s+in\s+mind)\s*(?:that\s+)?",
                      "", raw.strip(), flags=re.I).strip(" ,.")
        if len(fact.split()) < 2:
            return False
        self._add_note(f"Matt said: {fact}")
        self._eyes("HAPPY")
        self.body.speak("Got it. I'll remember that.")
        return True

    def _add_note(self, text):
        text = text.strip()[:200]
        words = set(re.findall(r"[a-z0-9']+", text.lower()))
        with self._notes_lock:
            for n in self.notes:
                old = set(re.findall(r"[a-z0-9']+", n.get("text", "").lower()))
                if words and len(words & old) / max(1, min(len(words), len(old))) >= 0.7:
                    return False
            self.notes.append({"text": text, "t": time.time(),
                               "day": datetime.date.today().isoformat()})
            self.notes = self.notes[-int(self.pcfg.get("max_notes", 40)):]
            self._save("notes.json", self.notes)
        _log(f"note: {text}")
        return True

    def conversation_over(self):
        """The follow-up window closed. Later, quietly, jot down anything
        worth remembering from it (resident model, one short request)."""
        if self.game and self.game["kind"] == "rps":
            self.game = None
        self.pending = None
        turns, self._turns = self._turns, 0
        if turns < 1 or not self._enabled("auto_notes") or self.brain is None:
            return
        try:
            recent = [t for t in list(self.memory.history)[-min(12, 2 * turns + 2):]
                      if t.get("content")]
        except Exception:
            return
        said = " ".join(t["content"] for t in recent if t.get("role") == "user")
        if len(said.split()) < 8:
            return
        if self._extract_timer is not None:
            self._extract_timer.cancel()
        self._extract_timer = threading.Timer(25, self._extract, args=(recent, time.time()))
        self._extract_timer.daemon = True
        self._extract_timer.start()

    def _extract(self, recent, ended_at):
        if self.last_interaction > ended_at:
            return   # Matt is talking again; the next close covers these turns
        with self._notes_lock:
            known = "\n".join(f"- {n['text']}" for n in self.notes[-20:]) or "(none)"
        convo = "\n".join(("Matt: " if t["role"] == "user" else "Spark: ") + t["content"][:400]
                          for t in recent)
        out = self._ask_brain(
            "You keep long-term notes about Matt for Spark, his desk robot. From the conversation, "
            "extract at most 2 NEW durable facts about Matt's own life worth remembering in future "
            "chats: plans, projects, upcoming events (with the date; today is "
            f"{datetime.date.today():%A, %B %d, %Y}), people, pets, likes and dislikes. Ignore "
            "general-knowledge questions, weather, lights, timers, games and small talk, and "
            "anything already in the notes. One short third-person line each, starting with '- '. "
            "If there is nothing new, reply NONE.\nExisting notes:\n" + known,
            convo, timeout=20)
        if not out or out.strip().upper().startswith("NONE"):
            return
        for line in out.splitlines()[:4]:
            line = line.strip()
            if line.startswith("- ") and 4 <= len(line.split()) <= 40:
                self._add_note(line[2:])

    # ------------------------------------------------------------ brain hooks
    def context(self):
        """System-prompt additions: notes, game on, late-night drowsiness."""
        out = ""
        with self._notes_lock:
            notes = list(self.notes[-15:])
        if notes:
            out += ("\nTHINGS YOU REMEMBER ABOUT MATT (your notes from past chats, with the day "
                    "noted; bring one up only when it fits naturally, never recite the list): "
                    + "; ".join(f"{n['text']} ({n.get('day', '')})" for n in notes))
        g = self.game
        if g and g["kind"] == "chat":
            topic = g.get("topic")
            if g.get("ending"):
                out += (f"\nGAME: Matt is ending your game of {g['name']}. Give the final score "
                        "if there was one and a short, warm sign-off.")
            elif g["name"] == "trivia":
                switched, g["switched"] = g.get("switched"), False
                out += ("\nGAME ON: you're hosting trivia for Matt"
                        + (f", category: {topic}" if topic else ", mixed categories") + ". "
                        + (f"He just switched the category to {topic}: say so in a few words and "
                           "ask your first question in it. " if switched else "")
                        + "Every turn: if he just answered, say right or wrong in a few words (give "
                          "the real answer when wrong) and the score, then ALWAYS ask the next "
                          "question. Never end a turn without a question: the game keeps going "
                          "until Matt says stop. One question at a time, never repeat one, mix "
                          "easy and hard. He can change the category any time.")
            else:
                out += (f"\nGAME ON: you're playing {g['name']} with Matt. Host it like a playful "
                        "friend: one question or turn at a time, then wait for his answer. React to "
                        "each answer (say if it's right and give the real answer), keep a running "
                        "score, then go straight to the next turn. Under 40 words per turn.")
        if self.night():
            out += "\n(It's late at night: you're a little drowsy, though still happy to help.)"
        back, self._back_after = self._back_after, None
        if back and time.time() - back[0] < 120:
            out += (f"\n(Matt just got back after about {int(back[1] // 60)} minutes away: open "
                    "with a quick, happy welcome back before answering.)")
        return out

    def turn_rule(self, nudge=False):
        """The per-turn reply rule while a question game is on: the default
        'answer only what was asked' stopped trivia after one question."""
        g = self.game
        if not (g and g["kind"] == "chat" and not g.get("ending") and g["name"] in _ASKING_GAMES):
            return None
        if nudge:   # her own prompt for the missing question: nothing to judge
            return " Game turn: don't judge or score anything, just ask your next question. At most 30 words."
        return (" Game turn: react to his answer in a few words, then ask your next "
                "question. The reply must end with that question. At most 45 words.")

    def wants_next(self, reply):
        """A question game turn ended without the next question."""
        return bool(self.turn_rule() and reply and "?" not in reply)

    def after_reply(self, reply):
        if reply:
            self._turns += 1   # a stored exchange: worth a look for notes
        g = self.game
        if not g or g["kind"] != "chat":
            return
        g["at"] = time.time()
        if g.get("ending"):
            self.game = None
            return
        head = (reply or "")[:80].lower()
        if re.search(r"^(?:yes|yep|right|correct|bingo)\b|\b(?:correct|that'?s\s+right|"
                     r"you'?re\s+right|you\s+got\s+it|nailed\s+it)\b", head) \
                and not re.search(r"\bnot\s+(?:quite\s+)?(?:right|correct)\b", head):
            self._eyes("OVERJOYED")
            self._arms(170, 20, speed=100)     # cheer
        elif re.search(r"\b(?:not quite|nope|wrong|close,? but|sorry,)\b", head):
            self._eyes("PUZZLED")
            self._arms((20, 120), 20, speed=60)
