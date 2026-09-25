"""Alarms, timers and reminders — persisted, cancellable, speakable.

Absolute-time alarms ("wake me at 7am") and relative timers ("in 10
minutes"), optionally labeled ("remind me to stretch"). State survives
restarts via state_dir/schedule.json: future items re-arm on boot,
expired ones are dropped with a log (a missed alarm replaying as a
surprise helps nobody).

Fire happens through the injected callback so the router owns speech;
threads are daemons so they never block shutdown.
"""
import json
import pathlib
import re
import sys
import threading
import time


def _log(msg):
    print(f"[sched] {msg}", file=sys.stderr)


def fmt_clock(h, m):
    """7, 30 -> '7:30 AM'."""
    mer = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {mer}"


def next_occurrence(hour, minute, meridiem, now=None):
    """Next datetime matching (hour, minute) at/after now. None if bogus.

    Meridiem rules: explicit am/pm honored ('in the morning' = am,
    'afternoon'/'evening' = pm); bare 7-11 assumes morning (wake alarms
    dominate), bare 1-6 assumes afternoon, bare 12 is noon.
    """
    import datetime as dt
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    h = hour
    low = (meridiem or "").lower()
    if "noon" in low:
        h, minute = 12, 0
    elif "midnight" in low:
        h, minute = 0, 0
    elif re.search(r"\ba\.?m\.?\b|morning", low):
        if h == 12:
            h = 0
    elif re.search(r"\bp\.?m\.?\b|afternoon|evening", low):
        if h < 12:
            h += 12
    elif h == 12:
        pass                                    # bare 12 -> noon
    elif 1 <= h <= 6:
        h += 12                                 # bare 1-6 -> afternoon
    now_dt = dt.datetime.now() if now is None else now
    target = now_dt.replace(hour=h, minute=minute, second=0, microsecond=0)
    if target <= now_dt:
        target += dt.timedelta(days=1)
    return target


class AlarmClock:
    def __init__(self, cfg, on_fire):
        """on_fire(kind, label) runs on the firing thread; keep it safe."""
        self.on_fire = on_fire
        sd = cfg.get("state_dir")
        self.state_path = pathlib.Path(sd) / "schedule.json" if sd else None
        self._items = {}       # id -> {"kind", "at", "label", "thread"}
        self._lock = threading.Lock()
        self._next = 1
        self._load()

    # ------------------------------------------------------------ public
    def add_timer(self, seconds, label=None, kind="timer"):
        seconds = max(1, int(seconds))
        return self._arm(kind, time.time() + seconds, label)

    def add_alarm(self, when, label=None):
        """when: epoch seconds (absolute)."""
        return self._arm("alarm", when, label)

    def cancel(self, kind="all"):
        """Cancel matching items; returns how many."""
        with self._lock:
            ids = [i for i, it in self._items.items() if kind in ("all", it["kind"])]
            for i in ids:
                it = self._items.pop(i)
                t = it.get("thread")
                if t:
                    t.cancel()
            self._save()
            return len(ids)

    def remaining(self, kind="timer"):
        """Seconds left on the soonest item of a kind, or None."""
        with self._lock:
            times = [it["at"] for it in self._items.values()
                     if it["kind"] == kind]
        if not times:
            return None
        return max(0, int(min(times) - time.time()))

    def status_lines(self):
        with self._lock:
            items = sorted(self._items.values(), key=lambda it: it["at"])
        out = []
        for it in items:
            left = max(0, int(it["at"] - time.time()))
            if it["kind"] == "alarm":
                lt = time.localtime(it["at"])
                out.append("alarm at " + fmt_clock(lt.tm_hour, lt.tm_min)
                           + (f" for {it['label']}" if it.get("label") else ""))
            elif it["kind"] == "reminder":
                out.append("reminder in " + self._human(left)
                           + (f": {it['label']}" if it.get("label") else ""))
            else:
                out.append(self._human(left) + " left"
                           + (f" — {it['label']}" if it.get("label") else ""))
        return out

    # ------------------------------------------------------------ engine
    def _arm(self, kind, at, label):
        iid = self._next
        self._next += 1
        delay = max(0.5, at - time.time())
        t = threading.Timer(delay, self._fire, args=[iid])
        t.daemon = True
        with self._lock:
            self._items[iid] = {"kind": kind, "at": at, "label": label,
                                "thread": t}
            self._save()
        t.start()
        return iid

    def _fire(self, iid):
        with self._lock:
            it = self._items.pop(iid, None)
            self._save()
        if not it:
            return
        try:
            self.on_fire(it["kind"], it.get("label"))
        except Exception as e:
            _log(f"fire callback failed: {e}")

    @staticmethod
    def _human(secs):
        if secs < 60:
            return f"{int(secs)} second" + ("s" if secs != 1 else "")
        if secs < 3600:
            m = secs / 60
            return (f"{int(m)}" if m == int(m) else f"{round(m, 1)}") + " minute" + \
                   ("" if m == 1 else "s")
        h = secs / 3600
        return (f"{int(h)}" if h == int(h) else f"{round(h, 1)}") + " hour" + \
               ("" if h == 1 else "s")

    # -------------------------------------------------------- persistence
    def _save(self):
        if not self.state_path:
            return
        try:
            data = [{"kind": it["kind"], "at": it["at"], "label": it.get("label")}
                    for it in self._items.values()]
            self.state_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except Exception as e:
            _log(f"save failed: {e}")

    def _load(self):
        if not (self.state_path and self.state_path.exists()):
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"load failed: {e}")
            return
        now = time.time()
        rearm = dropped = 0
        for it in data:
            if it.get("at", 0) <= now + 0.5:
                dropped += 1
                continue
            self._arm(it["kind"], it["at"], it.get("label"))
            rearm += 1
        if data:
            _log(f"restored {rearm}, dropped {dropped} expired")
