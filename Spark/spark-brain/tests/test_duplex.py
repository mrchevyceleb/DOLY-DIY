"""Interruption safety: echo, cancellation, handoff, and paused requests."""
import os
from pathlib import Path
import struct
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark.duplex import DuplexAudio, TurnInterrupted, interruptible, is_echo
from spark.ear import MicStream, record_utterance
from spark.streamspeech import PCMPlayer

CFG = {"audio": {"sample_rate": 16000, "input_device": "test", "echo_cancellation": False,
                 "silence_ms": 700, "max_utterance_ms": 15000, "hard_utterance_ms": 20000,
                 "start_rms": 900, "stop_rms": 500}}


class DuplexTests(unittest.TestCase):
    def test_distorted_wait_needs_raw_mic_agreement_but_real_stop_works(self):
        for raw_text, clean_text, interrupted in (
                ("Right now it's", "Wait.", False),
                ("Stop.", "Stop.", True),
                ("Hmm. Stop talking.", "Stop talking.", True),
                ("Actually turn down the volume.", "Actually turn", True)):
            with patch("spark.ear._speech_detector", return_value=None):
                duplex = DuplexAudio(CFG, lambda pcm: raw_text if pcm == b"raw" else clean_text)
            duplex.begin()
            duplex.expected = "Right now it's 67 degrees."
            duplex._verify(b"clean", duplex.epoch, duplex.expected, b"raw")
            self.assertEqual(duplex.cancel.is_set(), interrupted, raw_text)

    def test_weather_numbers_and_new_output_during_asr_are_echo(self):
        reply = "Right now it's 67 degrees, feels like 65. Clear sky. High of 67 today."
        for fragment in ("It's sixty seven degrees and", "High at sixty seven.",
                         "It's sixty-seven degrees.", "Clear sky.", "And sixty seven."):
            self.assertTrue(is_echo(fragment, reply), fragment)
        self.assertFalse(is_echo("actually volume sixty seven", reply))
        with patch("spark.ear._speech_detector", return_value=None):
            duplex = DuplexAudio(CFG, lambda clip: "sixty seven degrees")
        duplex.begin()
        duplex.expected = reply
        duplex._verify(b"test", duplex.epoch, "I will check.")
        self.assertFalse(duplex.cancel.is_set())

    def test_echo_is_rejected_but_a_new_command_about_same_subject_is_allowed(self):
        reply = "The lights are at fifty percent. Would you like them brighter?"
        self.assertTrue(is_echo("at fifty percent", reply))
        self.assertTrue(is_echo("would you like them bright", reply))
        self.assertFalse(is_echo("actually dim the lights", reply))
        self.assertFalse(is_echo("stop", reply))
        self.assertFalse(is_echo("stop", "Say stop if you'd like me to stop."))
        for phrase in ("please stop", "stop now", "please wait", "hold on", "stop it", "stop that", "wait a second",
                       "can you stop", "could you stop please", "stop the music", "stop now please"):
            self.assertFalse(is_echo(phrase, "You can say " + phrase + " any time."))

    def test_cancel_returns_while_network_is_stalled_and_closes_when_it_wakes(self):
        release, closed, cancel = threading.Event(), threading.Event(), threading.Event()
        def stalled():
            try:
                release.wait(1)
                yield "late reply"
            finally:
                closed.set()
        stream = interruptible(stalled(), cancel)
        threading.Timer(.05, cancel.set).start()
        started = time.monotonic()
        with self.assertRaises(TurnInterrupted):
            next(stream)
        self.assertLess(time.monotonic() - started, .2)
        release.set()
        self.assertTrue(closed.wait(1))

    def test_verified_interruption_retains_onset_and_discards_only_replayed_frames(self):
        with patch("spark.ear._speech_detector", return_value=None):
            duplex = DuplexAudio(CFG, lambda clip: "actually dim the lights")
        duplex.begin()
        duplex.history.extend([(1.0, b"onset"), (1.1, b"speech")])
        duplex._verify(b"test", duplex.epoch, "Today's weather is sunny.")
        self.assertTrue(duplex.cancel.is_set())
        self.assertEqual(duplex.take_pending(), (b"onsetspeech", 1.1))
        mic = MicStream(CFG)
        mic._queue.extend([(1.0, b"onset"), (1.1, b"speech"), (1.2, b"tail")])
        mic.discard_before(1.1)
        self.assertEqual(list(mic._queue), [(1.2, b"tail")])
        duplex.begin()
        duplex._verify(b"test", duplex.epoch-1, "stop")
        self.assertFalse(duplex.cancel.is_set())  # stale ASR cannot cancel next turn

    def test_playback_cancel_reaps_player_without_waiting_for_queued_audio(self):
        cancel = threading.Event()
        process = Mock()
        process.poll.return_value = None
        player = PCMPlayer(24000, {}, Mock(), cancel, Mock())
        with patch("subprocess.Popen", return_value=process):
            player.write(bytes(960))
            cancel.set()
            with self.assertRaises(TurnInterrupted):
                player.write(bytes(960))
            player.close()
        process.terminate.assert_called_once()
        self.assertEqual(process.stdin.write.call_count, 1)

    def test_long_request_with_a_half_second_pause_is_one_turn(self):
        voice = struct.pack("<320h", *([3000]*320))
        source = Mock(noise_floor=400, prefix_frames=0)
        source.frames = lambda: iter([voice]*250 + [bytes(640)]*25 + [voice]*250 + [bytes(640)]*40)
        with patch("spark.ear._speech_detector", return_value=None):
            pcm = record_utterance(source, CFG)
        self.assertGreater(len(pcm)/32000, 10)
        self.assertIn(voice*250 + bytes(640)*25 + voice*250, pcm)

    def test_near_voice_pauses_before_slow_verification_and_echo_resumes(self):
        release, requested = threading.Event(), threading.Event()
        verified_sizes = []
        def slow_asr(clip):
            verified_sizes.append(len(clip))
            requested.set()
            release.wait(1)
            return "the lights are on"
        with patch("spark.ear._speech_detector", return_value=Mock(is_speech=lambda *args: True)):
            duplex = DuplexAudio(CFG, slow_asr)
        duplex.echo = Mock(process=lambda pcm, stamp: pcm, last_render=0.0, rendered_frames=100)
        duplex.begin(120)
        duplex.expected = "The lights are on."
        voice = struct.pack("<320h", *([3000]*320))
        for n in range(80):
            duplex.capture(bytes(640), 98+n*.02)
        for n in range(8):
            duplex.capture(voice, 100+n*.02)
        self.assertTrue(duplex.pause.is_set())
        self.assertFalse(duplex.cancel.is_set())
        for n in range(8, 26):
            duplex.capture(voice, 100+n*.02)
        self.assertTrue(requested.wait(.3))
        self.assertEqual(verified_sizes, [64000, 64000])  # one raw and one cleaned check
        self.assertLess(len(duplex.history), 40)  # handoff keeps short onset
        release.set()
        for _ in range(100):
            if not duplex.checking:
                break
            time.sleep(.005)
        self.assertFalse(duplex.pause.is_set())
        self.assertFalse(duplex.cancel.is_set())


if __name__ == "__main__":
    unittest.main()
