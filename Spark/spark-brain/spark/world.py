"""Matt's world: calendar, tasks, mail and server status from TARDIS on Moria.

Read-only. A turn that asks about one of these gets the live data as tool
context, the same way a weather question gets the forecast. Nothing is
fetched for ordinary chat.
"""
import datetime
import json
import re
import sys
import threading
import time
import urllib.request

TAG = "MATT'S LIVE DATA"

_TOPICS = {
    "calendar": re.compile(
        r"\b(?:calendar|schedule|agenda|meetings?|appointments?|"
        r"(?:am i|are we) (?:free|busy)|"
        r"what(?:'s| is| do i have| have i got) (?:on |going on |happening |coming up )?"
        r"(?:today|tomorrow|tonight|this week|next week)|"
        r"(?:anything|something|what'?s) (?:on |planned |coming up )(?:for )?"
        r"(?:today|tomorrow|tonight|this week|next week))\b", re.I),
    "tasks": re.compile(
        r"\b(?:tasks?|to-?do(?: list)?s?|overdue|on my plate|"
        r"what should i (?:work on|do (?:today|next|first)))\b", re.I),
    "mail": re.compile(
        r"\b(?:e-?mails?|inbox|(?:any|new|unread|my) (?:mail|messages))\b", re.I),
    "server": re.compile(
        r"\b(?:moria|tardis|(?:my|the) (?:server|agents?)|background (?:jobs?|work))\b", re.I),
}
# 'brief me', 'how's my day looking': the whole picture in one go
_BRIEF_RE = re.compile(
    r"\b(?:brief(?:ing)? me|(?:daily|morning) brief(?:ing)?|catch me up|status report|"
    r"(?:how'?s|how is|what'?s|what is) my (?:day|week)(?: look(?:ing)?| like)?)\b", re.I)
# 'and tomorrow?', 'what about Friday?': a follow-up that leans on the turn before
_FOLLOWUP_RE = re.compile(
    r"^\W*(?:and|or|what about|how about|what else|anything else|which one)\W|"
    r"(?:after that|the day after|the next one|the first one|read (?:it|them|that)|"
    r"tell me more)", re.I)
# a bare day is a follow-up only on its own ('Friday?', 'and this weekend?'),
# never inside a sentence ('how are you today?')
_DAY_RE = re.compile(
    r"(?:tomorrow|tonight|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"this week|next week|weekend)", re.I)


def _follows(text):
    text = (text or "").strip()
    if _FOLLOWUP_RE.search(text):
        return True
    return bool(_DAY_RE.search(text)) and len(text.split()) <= 3
# her own timers and alarms belong to the router, never to the calendar
_LOCAL_RE = re.compile(r"\b(?:timers?|alarms?|remind me)\b", re.I)


def _log(msg):
    print(f"[world] {msg}", file=sys.stderr)


def topics(text):
    """Which of Matt's data a sentence asks about (possibly none)."""
    text = text or ""
    if _LOCAL_RE.search(text):
        return set()
    if _BRIEF_RE.search(text):
        return {"calendar", "tasks", "mail"}
    return {name for name, rx in _TOPICS.items() if rx.search(text)}


def _clock(dt):
    return dt.strftime("%I:%M %p").lstrip("0").replace(":00", "")


def _day(date, today):
    delta = (date - today).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    return date.strftime("%A, %B ") + str(date.day)


def _short(text, limit=80):
    # mail subjects carry zero-width padding the synth would trip on
    text = "".join(ch for ch in str(text or "") if ch.isprintable() or ch.isspace())
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


class World:
    def __init__(self, cfg):
        wcfg = cfg.get("world", {}) or {}
        self.url = (wcfg.get("url") or "").rstrip("/")
        self.enabled = bool(wcfg.get("enabled", True) and self.url)
        self.timeout = wcfg.get("timeout_s", 6)
        self._cache = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- fetch
    def _get(self, path):
        """Parsed JSON for one TARDIS route, cached two minutes; None on failure."""
        with self._lock:
            hit = self._cache.get(path)
        # agent and job counts move in seconds; the rest of his day does not
        ttl = 20 if path in ("/api/health", "/api/summary") else 120
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        try:
            req = urllib.request.Request(self.url + path, headers={"Accept": "application/json"})
            data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
        except Exception as e:
            _log(f"{path} failed: {e}")
            return None
        with self._lock:
            self._cache[path] = (time.time(), data)
        return data

    # --------------------------------------------------------------- render
    def _calendar(self):
        data = self._get("/api/calendar")
        if not isinstance(data, dict):
            return None
        now = datetime.datetime.now().astimezone()
        today = now.date()
        days = {}   # date -> what is on it, in order

        def starts(ev):
            # the absolute instant: raw strings misorder across UTC offsets
            try:
                start = ev.get("start") or {}
                if start.get("dateTime"):
                    return datetime.datetime.fromisoformat(start["dateTime"]).timestamp()
                day = datetime.date.fromisoformat(start["date"])
                return datetime.datetime.combine(day, datetime.time()).astimezone().timestamp()
            except (AttributeError, KeyError, TypeError, ValueError):
                return float("inf")
        for ev in sorted(data.get("events") or [], key=starts):
            try:
                start, end = ev.get("start") or {}, ev.get("end") or {}
                title = _short(ev.get("summary") or "Untitled event")
                if start.get("dateTime"):
                    begins = datetime.datetime.fromisoformat(start["dateTime"]).astimezone()
                    ends = (datetime.datetime.fromisoformat(end["dateTime"]).astimezone()
                            if end.get("dateTime") else begins)
                    if ends < now or (begins.date() - today).days > 10:
                        continue
                    date = begins.date()
                    line = f"{_clock(begins)} to {_clock(ends)} {title}"
                    if begins <= now:
                        line += " (happening now)"
                else:
                    date = datetime.date.fromisoformat(start["date"])
                    if date < today or (date - today).days > 10:
                        continue
                    line = f"all day {title}"
                days.setdefault(date, []).append(line)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        if not days:
            return "CALENDAR: nothing on Matt's calendar in the next ten days."
        # One line per day, empty days included: asked 'and Friday?' with
        # only NEXT Friday booked, the brain answered with next week's event.
        def label(d):
            name = d.strftime("%A %B ") + str(d.day)
            gap = (d - today).days
            return (f"TODAY ({name})" if gap == 0 else f"TOMORROW ({name})" if gap == 1
                    else f"THIS {name}" if gap < 7 else f"NEXT WEEK {name}")
        span = [today + datetime.timedelta(days=n) for n in range(7)]
        span += sorted(d for d in days if d > span[-1])
        out = [f"{label(d)}: " + ("; ".join(days[d][:6]) if d in days else
                                  "nothing left" if d == today else "nothing")
               for d in span]
        return ("CALENDAR, one line per day. A weekday said alone ('Friday') means the "
                "THIS line, never the NEXT WEEK line:\n" + "\n".join(out))

    def _tasks(self):
        data = self._get("/api/tasks")
        if not isinstance(data, list):
            return None
        open_tasks = [t for t in data if isinstance(t, dict) and t.get("status") != "done"]
        rank = {"high": 0, "medium": 1, "low": 2}
        open_tasks.sort(key=lambda t: rank.get(t.get("priority"), 1))

        def names(due, cap):
            picked = [t for t in open_tasks if due(t.get("due"))]
            out = "; ".join(_short(t.get("title"), 70) for t in picked[:cap])
            return len(picked), out + (f"; and {len(picked) - cap} more" if len(picked) > cap else "")

        n_today, today = names(lambda d: d == "today", 6)
        n_over, overdue = names(lambda d: d == "overdue", 5)
        n_next, upcoming = names(lambda d: d not in ("today", "overdue"), 4)
        out = (f"TASKS: {len(open_tasks)} open, {n_today} due today, {n_over} overdue. "
               f"Due today: {today or 'none'}.")
        if n_over:
            out += f" Overdue, most important first: {overdue}."
        if n_next:
            out += f" Coming up: {upcoming}."
        return out

    def _mail(self):
        data = self._get("/api/email")
        if not isinstance(data, list):
            return None
        waiting = [m for m in data if isinstance(m, dict)
                   and (m.get("unread") or m.get("status") == "needs_reply")]
        if not waiting:
            return "MAIL: nothing unread or waiting on a reply."
        def age(m):
            a = str(m.get("age") or "")
            return f" ({a} ago)" if a[:1].isdigit() else f" ({a})" if a else ""
        items = "; ".join(f"{_short(m.get('from'), 40)}: {_short(m.get('subject'), 70)}{age(m)}"
                          for m in waiting[:6])
        more = f"; and {len(waiting) - 6} more" if len(waiting) > 6 else ""
        return f"MAIL: {len(waiting)} unread or waiting on a reply, newest first: {items}{more}"

    def _server(self):
        health = self._get("/api/health")
        summary = self._get("/api/summary")
        if not isinstance(health, dict):
            return None
        out = (f"SERVER (TARDIS on Moria): online, {health.get('busyTurns', 0)} agent turns "
               f"running, {health.get('backgroundTasks', 0)} background tasks.")
        if isinstance(summary, dict):
            out += (f" Jobs: {summary.get('runningJobs', 0)} running, "
                    f"{summary.get('queuedJobs', 0)} queued, "
                    f"{summary.get('needsReview', 0)} waiting on Matt's review.")
        return out

    # -------------------------------------------------------------- context
    def wanted(self, text, last_topics=None):
        """Topics this turn asks about. A short follow-up that leans on the
        turn before ('and tomorrow?') keeps that turn's topics."""
        if not self.enabled:
            return set()
        found = topics(text)
        if (not found and last_topics and len((text or "").split()) <= 8
                and _follows(text) and not _LOCAL_RE.search(text or "")):
            found = set(last_topics)
        return found

    def context(self, wanted):
        """Tool-context block holding the live data for those topics."""
        order = [t for t in ("calendar", "tasks", "mail", "server") if t in wanted]
        results = {}

        def fetch(topic):
            results[topic] = getattr(self, "_" + topic)()
        threads = [threading.Thread(target=fetch, args=(t,), daemon=True) for t in order]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=self.timeout + 2)
        parts = [results.get(t) or f"{t.upper()}: could not be reached just now. "
                                    "Say so plainly; never guess at it." for t in order]
        _log(f"context for {order}: {sum(len(p) for p in parts)} chars")
        return (f"{TAG} from his own server, fetched just now. Answer from this and nothing "
                "else: say the next or most important thing first, give counts instead of "
                "reading long lists aloud, and never invent an event, task or email.\n"
                + "\n".join(parts))
