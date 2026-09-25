"""Alarms, timers, reminders: parsing, routing, persistence."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark import commands as cmds
from spark.router import Router
from spark.sched import AlarmClock, fmt_clock, next_occurrence

CFG = {"state_dir": None}


def _router(alarms=None):
    r = Router.__new__(Router)
    r.cfg = dict(CFG)
    r.body = Mock()
    r.brain = Mock()
    r.memory = Mock()
    r.llm_reply = None
    r.last_motion_result = None
    r._last_motion_at = 0
    r._timers = []
    r.govee = None
    r.alarms = alarms if alarms is not None else Mock(spec=AlarmClock)
    r.alarms._human = lambda secs: AlarmClock._human(secs)
    return r


class TimeParsingTests(unittest.TestCase):
    def test_clock_times(self):
        self.assertEqual(cmds.parse_clock_time("set an alarm for 7:30 am"), (7, 30, "am"))
        self.assertEqual(cmds.parse_clock_time("wake me at seven thirty"), (7, 30, None))
        self.assertEqual(cmds.parse_clock_time("alarm at 6 pm"), (6, 0, "pm"))
        self.assertEqual(cmds.parse_clock_time("set an alarm at 19 45"), (19, 45, None))
        self.assertEqual(cmds.parse_clock_time("noon"), (12, 0, "noon"))
        self.assertEqual(cmds.parse_clock_time("midnight"), (0, 0, "midnight"))
        self.assertIsNone(cmds.parse_clock_time("remind me tomorrow"))

    def test_next_occurrence_rules(self):
        import datetime as dt
        now = dt.datetime(2026, 9, 25, 20, 0)          # 8 PM
        self.assertEqual(next_occurrence(7, 30, "am", now), dt.datetime(2026, 9, 26, 7, 30))
        self.assertEqual(next_occurrence(11, 0, "pm", now), dt.datetime(2026, 9, 25, 23, 0))
        self.assertEqual(next_occurrence(6, 0, None, now), dt.datetime(2026, 9, 26, 18, 0))  # bare 6 -> pm
        self.assertEqual(next_occurrence(9, 0, None, now), dt.datetime(2026, 9, 26, 9, 0))   # bare 9 -> am
        self.assertEqual(next_occurrence(12, 0, "noon", now), dt.datetime(2026, 9, 26, 12, 0))
        self.assertIsNone(next_occurrence(99, 0, None, now))

    def test_fmt_clock(self):
        self.assertEqual(fmt_clock(7, 5), "7:05 AM")
        self.assertEqual(fmt_clock(12, 0), "12:00 PM")
        self.assertEqual(fmt_clock(0, 30), "12:30 AM")


class RouterSchedTests(unittest.TestCase):
    def _say(self, text, alarms):
        r = _router(alarms)
        self.assertTrue(r.handle(text))
        return r.body.speak.call_args[0][0]

    def test_set_alarm(self):
        a = Mock(spec=AlarmClock)
        self.assertIn("For what time", self._say("set an alarm", a))
        self.assertIn("Alarm set for", self._say("set an alarm for 7 am", a))
        a.add_alarm.assert_called_once()
        self.assertTrue(a.add_alarm.call_args.kwargs["label"].startswith("7"))

    def test_cancel_and_query(self):
        a = Mock(spec=AlarmClock)
        a.cancel.return_value = 1
        a.status_lines.return_value = ["alarm at 7:30 AM"]
        self.assertEqual(self._say("cancel my alarm", a), "Alarm cancelled.")
        self.assertIn("7:30", self._say("what alarms do I have", a))
        a.cancel.return_value = 0
        self.assertIn("don't have", self._say("cancel my alarm", a))

    def test_reminder_labeled(self):
        a = Mock(spec=AlarmClock)
        reply = self._say("remind me to stretch in ten minutes", a)
        self.assertIn("stretch", reply)
        a.add_timer.assert_called_once()
        self.assertEqual(a.add_timer.call_args.kwargs.get("label"), "stretch")
        self.assertEqual(a.add_timer.call_args.kwargs.get("kind"), "reminder")

    def test_timer_cancel_and_query(self):
        a = Mock(spec=AlarmClock)
        a.cancel.return_value = 1
        a.remaining.return_value = 95
        a._human = lambda s: "2 minutes"
        self.assertEqual(self._say("cancel the timer", a), "Timer cancelled.")
        self.assertIn("left on your timer", self._say("how much time is left on the timer", a))


class PendingSlotTests(unittest.TestCase):
    """Her own question must be answered by the next utterance (16:25 log)."""

    def _router(self, alarms):
        r = _router(alarms)
        r._pending = None
        return r

    def test_timer_answer_fills_slot(self):
        a = Mock(spec=AlarmClock)
        r = self._router(a)
        self.assertTrue(r.handle("set a timer"))
        self.assertIn("How long", r.body.speak.call_args[0][0])
        self.assertEqual(r._pending["kind"], "timer")
        a.add_timer.assert_not_called()
        self.assertTrue(r.handle("thirty seconds"))
        a.add_timer.assert_called_once_with(30)
        self.assertIn("30 seconds", r.body.speak.call_args[0][0])
        self.assertIsNone(r._pending)

    def test_alarm_answer_fills_slot(self):
        a = Mock(spec=AlarmClock)
        r = self._router(a)
        self.assertTrue(r.handle("set an alarm"))
        self.assertEqual(r._pending["kind"], "alarm")
        self.assertTrue(r.handle("7 am"))
        a.add_alarm.assert_called_once()
        self.assertIn("Alarm set", r.body.speak.call_args[0][0])

    def test_moving_on_clears_slot(self):
        a = Mock(spec=AlarmClock)
        r = self._router(a)
        r.handle("set a timer")
        a.reset_mock()
        self.assertTrue(r.handle("what is the weather"))    # routes to weather
        self.assertIsNone(r._pending)                        # slot forgotten
        a.add_timer.assert_not_called()

class CorrectionTests(unittest.TestCase):
    """'no, thirty seconds' right after a misheard-unit set re-sets it."""

    def test_timer_correction(self):
        a = Mock(spec=AlarmClock)
        r = _router(a)
        r._pending = r._last_set = None
        self.assertTrue(r.handle("set a timer for thirty"))     # ASR ate 'seconds'
        a.add_timer.assert_called_once_with(1800)               # read as minutes
        self.assertTrue(r.handle("no, thirty seconds"))
        a.cancel.assert_called_once_with("timer")
        self.assertEqual(a.add_timer.call_args[0][0], 30)
        self.assertIn("30 seconds", r.body.speak.call_args[0][0])

    def test_alarm_correction(self):
        a = Mock(spec=AlarmClock)
        r = _router(a)
        r._pending = r._last_set = None
        self.assertTrue(r.handle("set an alarm for 7 am"))
        a.add_alarm.assert_called_once()
        self.assertTrue(r.handle("actually, make it 8 am"))
        a.cancel.assert_called_once_with("alarm")
        self.assertEqual(a.add_alarm.call_count, 2)

    def test_plain_agreement_does_not_correct(self):
        a = Mock(spec=AlarmClock)
        r = _router(a)
        r._pending = r._last_set = None
        r.handle("set a timer for five minutes")
        a.reset_mock()
        self.assertTrue(r.handle("what time is it"))
        a.cancel.assert_not_called()
        a.add_timer.assert_not_called()

class CelebrationTests(unittest.TestCase):
    def _router(self):
        r = _router()
        r.cfg = {"state_dir": None, "alerts": {"celebrate": True,
                                               "celebrate_max_s": 1,
                                               "party_lights": False}}
        r.body.actuators_held = Mock(return_value=True)
        r.body.docked = False
        r.body.battery_pct = Mock(return_value=80)
        return r

    def test_timer_fire_celebrates_until_stopped(self):
        r = self._router()
        r._scheduled_fire("timer", None)
        ev = r._celebration
        self.assertIsNotNone(ev)
        import time as _t
        _t.sleep(0.3)
        r._stop_celebration()
        _t.sleep(0.3)
        spoken = [c[0][0] for c in r.body.speak.call_args_list]
        self.assertTrue(any("celebrate" in str(x).lower() for x in spoken))
        r.body.arms_party.assert_called()
        r.body.dance.assert_not_called()          # actuators held -> arms only

    def test_any_utterance_ends_the_party(self):
        r = self._router()
        r._scheduled_fire("alarm", "7:30 AM")
        self.assertIsNotNone(r._celebration)
        self.assertTrue(r.handle("what time is it"))
        import time as _t
        _t.sleep(0.3)
        self.assertIsNone(r._celebration)

    def test_reminder_stays_spoken_only(self):
        r = self._router()
        r._scheduled_fire("reminder", "stretch")
        self.assertIsNone(getattr(r, "_celebration", None))
        spoken = [c[0][0] for c in r.body.speak.call_args_list]
        self.assertTrue(any("stretch" in str(x) for x in spoken))

class AlarmClockTests(unittest.TestCase):
    def test_fire_and_persistence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            fired = []
            ck = AlarmClock({"state_dir": td}, lambda kind, label: fired.append((kind, label)))
            ck.add_timer(1, label="stretch", kind="reminder")
            ck.add_alarm(time.time() + 3600, label="7:30 AM")
            self.assertEqual(len(ck.status_lines()), 2)
            # reload from disk: future alarm re-arms, fired timer is gone
            deadline = time.time() + 4
            while time.time() < deadline and not fired:
                time.sleep(0.1)
            self.assertIn(("reminder", "stretch"), fired)
            ck2 = AlarmClock({"state_dir": td}, lambda kind, label: fired.append((kind, label)))
            self.assertEqual(len(ck2.status_lines()), 1)   # fired reminder gone, alarm re-armed
            self.assertIn("alarm", ck2.status_lines()[0])

    def test_cancel_removes(self):
        ck = AlarmClock({"state_dir": None}, lambda k, l: None)
        ck.add_timer(60)
        self.assertEqual(ck.cancel("timer"), 1)
        self.assertEqual(ck.remaining("timer"), None)


if __name__ == "__main__":
    unittest.main()
