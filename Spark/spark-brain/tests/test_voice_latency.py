"""Checks for the observed streaming, capture and follow-up failures."""
import io
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark.body import Body
from spark.ear import CommandAudio, MicStream, WakeResult, listen_for_wake, record_utterance
from spark.__main__ import Spark

CFG = {"audio": {"input_device": "test", "sample_rate": 16000,
                 "silence_ms": 460, "max_utterance_ms": 5000,
                 "start_rms": 900, "stop_rms": 500, "wake_weak_rms": 4000}}


def pcm(level):
    return struct.pack("<320h", *([level] * 320))


class VoiceLatencyTests(unittest.TestCase):
    def body(self):
        b = Body({}, hw=False)
        b.has = {"tts": True, "sound": True}
        b._produce_speech = Mock()
        b._snd = Mock()
        b._wav_duration = lambda path: 5.0
        return b

    def test_stream_plays_before_requesting_next_sentence_and_cleans_up_on_error(self):
        b = self.body()

        def sentences():
            yield "First sentence."
            b._snd.play.assert_called_once()
            raise RuntimeError("network disconnected")

        with patch("shutil.copyfile"), patch("spark.body.os.remove") as remove, \
                patch("spark.body.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "network disconnected"):
                b.speak_stream(sentences())
            remove.assert_called_once()
            self.assertTrue(b.speaking_recently())

    def test_echo_guard_counts_playback_once_in_both_modes(self):
        for wait in (True, False):
            b = self.body()
            now = [100.0]
            with patch("spark.body.time.time", side_effect=lambda: now[0]), \
                    patch("spark.body.time.sleep", side_effect=lambda s: now.__setitem__(0, now[0]+s)):
                self.assertTrue(b.speak("Hello", wait=wait))
                self.assertEqual(b._speaking_until, 105.25)
                now[0] = 105.30
                self.assertFalse(b.speaking_recently())

    def test_capture_drains_without_a_consumer_and_bounds_backlog(self):
        mic = MicStream(CFG)
        mic.proc = Mock(stdout=io.BytesIO(b"".join(pcm(n) for n in range(100))))
        mic._capture()
        self.assertEqual(len(mic._queue), 50)
        self.assertEqual(mic._queue[-1][1], pcm(99))
        mic.discard()
        self.assertFalse(mic._queue)
        with self.assertRaisesRegex(RuntimeError, "microphone capture failed"):
            mic._next()
        mic.retain(14)
        mic.proc.stdout = io.BytesIO(b"".join(pcm(n) for n in range(200)))
        mic._capture()
        self.assertEqual(len(mic._queue), 200)  # ASR pause keeps the command start
        self.assertEqual(mic._queue[0][1], pcm(0))
        mic.retain(1)
        self.assertEqual(len(mic._queue), 50)

    def test_room_noise_endpoints_without_five_second_cap_and_keeps_onset(self):
        quiet, speech = pcm(1000), pcm(3000)
        source = Mock(noise_floor=1000)
        source.prefix_frames = 0
        source.frames = lambda: iter([quiet]*10 + [speech]*20 + [quiet]*100)
        recording = record_utterance(source, CFG)
        self.assertTrue(recording.startswith(quiet))
        self.assertIn(speech*20, recording)
        self.assertLess(len(recording)/32000, 1.2)
        source.prefix_frames = 200
        source.frames = lambda: iter([speech]*600)
        self.assertLessEqual(len(record_utterance(source, CFG))/32000, 8)

    def test_strong_partial_wakes_early_and_keeps_consumed_audio(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = "spark"
        result = listen_for_wake(iter([pcm(3000)]*100), rec, CFG, ["spark"])
        self.assertTrue(result.prefix_pcm)
        self.assertLess(len(result.prefix_pcm)/32000, 0.5)
        rec.finish.assert_not_called()
        rec.partial.return_value = "park"
        self.assertFalse(listen_for_wake(iter([pcm(5000)]*20), rec, CFG, ["spark"]))

    def test_early_wake_only_segment_retries_and_preserves_one_word_command(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "asr": {"server_url": "test"}}
        spark.whisper = Mock()
        now = [100.0]
        responses = iter(["Spark.", "Stop."])
        def delayed_asr(_):
            now[0] += 10  # longer than the six-second onset window
            return next(responses)
        spark.whisper.transcribe_pcm.side_effect = delayed_asr
        rec = Mock()
        mic = Mock(noise_floor=500)
        mic.frames = lambda: iter([])
        with patch("spark.ear.record_utterance", return_value=pcm(3000)*30) as capture, \
                patch("spark.__main__.time.monotonic", side_effect=lambda: now[0]), \
                patch("spark.__main__.time.perf_counter", side_effect=lambda: now[0]):
            text, _ = spark._listen_command(mic, rec, WakeResult("spark", pcm(3000)))
        self.assertEqual(text, "Stop.")
        self.assertIsNone(capture.call_args.kwargs["on_frame"])
        rec.feed.assert_not_called()
        text, _ = spark._listen_command(Mock(), Mock(), WakeResult("spark stop"))
        self.assertEqual(text, "stop")

    def test_pause_in_wake_prefix_keeps_remaining_command_and_tts_cannot_raise_floor(self):
        mic = MicStream(CFG)
        mic._levels.extend([600]*50)
        mic.learn_noise(False)
        mic.proc = Mock(stdout=io.BytesIO(pcm(10000)*100))
        mic._capture()
        self.assertEqual(mic.noise_floor, 600)
        source = Mock(noise_floor=600)
        source.frames = lambda: iter([pcm(0)]*50)
        name, command = pcm(3000)*15, pcm(5000)*20
        audio = CommandAudio(source, name + pcm(0)*30 + command)
        first = record_utterance(audio, CFG)
        second = record_utterance(audio, CFG)
        self.assertNotIn(pcm(5000), first)
        self.assertIn(command, second)

    def test_followup_gets_full_window_and_silence_returns_to_wake_without_nag(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "conversation": {"follow_up_window_s": 8}}
        spark.body = Mock(has={k: True for k in ("helper", "touch", "tts", "sound")})
        spark.body.speaking_recently.return_value = False
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        spark._wait_for_wake = Mock(side_effect=[WakeResult("spark stop"), StopIteration])
        spark._listen_command = Mock(side_effect=[("stop", b""), ("", b"")])
        spark.converse = Mock()
        mic = Mock()
        mic.__enter__ = Mock(return_value=mic)
        mic.__exit__ = Mock(return_value=False)
        with patch("spark.asr.Recognizer"), patch("spark.ear.MicStream", return_value=mic), \
                patch("spark.__main__.threading.Thread"), patch("spark.__main__.sd_notify"):
            with self.assertRaises(StopIteration):
                spark.voice_loop()
        self.assertEqual([c.kwargs["timeout_s"] for c in spark._listen_command.call_args_list], [6, 8])
        spark.body.wake_reaction.assert_called_once()
        spark.body.speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
