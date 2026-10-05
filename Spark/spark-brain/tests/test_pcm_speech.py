"""Checks for voice identity across packets and safe stream failures."""
import io
import os
from pathlib import Path
import struct
import sys
import time
import threading
import unittest
from email.message import Message
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0

from spark import streamspeech, voicefx
from spark.body import Body


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = "application/x-spark-pcm"


class PCMSpeechTests(unittest.TestCase):
    def test_speech_boost_preserves_volume_headroom_and_chunk_boundaries(self):
        quiet = struct.pack("<100h", *([3000]*100))
        loud = struct.pack("<100h", *([30000, -30000]*50))
        source = quiet + loud
        boosted = voicefx.speech_gain(source, 6)
        values = struct.unpack("<200h", boosted)
        self.assertGreater(values[0], 5900)
        self.assertLess(max(abs(v) for v in values), 31200)
        self.assertEqual(boosted, voicefx.speech_gain(quiet, 6)+voicefx.speech_gain(loud, 6))
        self.assertEqual(source, voicefx.speech_gain(source, 0))

    def test_packet_effects_match_chosen_whole_take(self):
        samples = struct.pack("<9600h", *[int(12000 * ((i % 31) / 31 - .5))
                                           for i in range(9600)])
        expected = voicefx._ring_mod(voicefx._pitch_shift(samples, 24000, .75), 24000, .16)
        effects = voicefx.PCMEffects(24000, .75, .16)
        actual = b"".join(effects.process(samples[i:i+640])
                          for i in range(0, len(samples), 640))
        self.assertEqual(actual, expected)

    def test_broken_stream_reports_audio_started_and_reaps_player(self):
        frame = struct.pack("<24000h", *([1000] * 24000))
        response = Response(b"SPK1" + struct.pack("<I", 24000)
                            + struct.pack("<I", len(frame)) + frame)  # missing EOS
        process = Mock(stdin=io.BytesIO())
        process.poll.return_value = None
        process.wait.return_value = 0
        cfg = {"server_url": "http://test", "voice_name": "warm-soft-robot"}
        guard = Mock()
        with patch("urllib.request.urlopen", return_value=response), \
                patch("subprocess.Popen", return_value=process):
            with self.assertRaises(streamspeech.StreamSpeechError) as failure:
                streamspeech.play("Hi Matt.", cfg, guard)
        self.assertTrue(failure.exception.played)
        process.terminate.assert_called_once()
        self.assertLessEqual(guard.call_args.args[0], time.time() + .26)

    def test_partial_audio_never_replays_through_fallback(self):
        body = Body({"tts": {"server_streaming": True, "server_url": "test"}}, hw=False)
        body.hw = True
        body.has = {"tts": True, "sound": True}
        body._produce_speech = Mock()
        with patch("spark.streamspeech.play", side_effect=streamspeech.StreamSpeechError("lost", True)):
            body.speak_stream(iter(["First sentence.", "Second sentence."]))
        body._produce_speech.assert_not_called()

    def test_failed_short_prebuffer_falls_back_without_playing_a_fragment(self):
        frame = struct.pack("<4800h", *([1000] * 4800))
        response = Response(b"SPK1" + struct.pack("<I", 24000)
                            + struct.pack("<I", len(frame)) + frame)  # missing EOS
        cfg = {"server_url": "http://test", "voice_name": "warm-soft-robot"}
        with patch("urllib.request.urlopen", return_value=response), patch("subprocess.Popen") as player:
            with self.assertRaises(streamspeech.StreamSpeechError) as failure:
                streamspeech.play("A complete thought.", cfg, Mock())
        self.assertFalse(failure.exception.played)
        player.assert_not_called()

    def test_prefetch_does_not_remember_an_unheard_or_interrupted_chunk(self):
        body = Body({"tts": {"server_streaming": True, "server_url": "test"}}, hw=False)
        body.hw = True
        body.has = {"tts": True, "sound": True}
        said = []
        with patch("spark.streamspeech.play", side_effect=[.1, streamspeech.StreamSpeechError("lost", True)]):
            body.speak_stream(iter(["First sentence.", "Second sentence.", "Unheard third."]), on_spoken=said.append)
        self.assertEqual(said, ["First sentence."])

    def test_prefetch_prepares_next_take_before_current_playback_finishes(self):
        body = Body({"tts": {"server_streaming": True, "server_url": "test"}}, hw=False)
        body.hw = True
        body.has = {"tts": True, "sound": True}
        second_ready = threading.Event()
        def fetch(prepared):
            if prepared.text == "Second sentence.":
                second_ready.set()
            prepared.done.set()
        def play(text, *args, **kwargs):
            if text == "First sentence.":
                self.assertTrue(second_ready.wait(.5))
            return .1
        with patch.object(streamspeech.PreparedPCM, "fetch", fetch), patch("spark.streamspeech.play", side_effect=play):
            body.speak_stream(iter(["First sentence.", "Second sentence."]))

    def test_failure_before_audio_goes_straight_to_fallback(self):
        body = Body({"tts": {"server_streaming": True, "server_url": "test"}}, hw=False)
        body.hw = True
        body.has = {"tts": True, "sound": True}
        body._snd = Mock()
        body._produce_speech = Mock()
        body._wav_duration = lambda _: 0
        with patch("spark.streamspeech.play", side_effect=streamspeech.StreamSpeechError("busy")), \
                patch("shutil.copyfile"), patch("spark.body.os.remove"), patch("spark.body.time.sleep"):
            body.speak_stream(iter(["First sentence."]))
        body._produce_speech.assert_called_once_with("First sentence.", skip_primary=True)


if __name__ == "__main__":
    unittest.main()
