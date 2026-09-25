"""Checks for the observed streaming, capture and follow-up failures."""
import io
import math
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
from spark.ear import CommandAudio, MicStream, SpeechHighPass, WakeResult, listen_for_wake, record_utterance
from spark.__main__ import Spark

CFG = {"audio": {"input_device": "test", "sample_rate": 16000,
                 "silence_ms": 460, "max_utterance_ms": 5000,
                 "start_rms": 900, "stop_rms": 500, "wake_weak_rms": 3000,
                 "highpass_hz": 0}}


def pcm(level):
    return struct.pack("<320h", *([level] * 320))


class VoiceLatencyTests(unittest.TestCase):
    def test_chat_receives_live_state_instead_of_assuming_a_charger(self):
        import json
        spark = Spark.__new__(Spark)
        spark.body = Mock(docked=False, has={"edge": True})
        spark.body.refresh_power.return_value = False
        spark.body.battery_pct.return_value = 42
        spark.body._edge_gaps.return_value = []
        spark.router = Mock(last_motion_result={"command": "come_here", "result": "ambiguous"})
        context = spark._body_context()
        state = json.loads(context.split(": ", 1)[1].split("\n", 1)[0])
        self.assertEqual(state["charging"], False)
        self.assertEqual(state["dock_motor_hold"], False)
        self.assertEqual(state["last_movement_result"]["result"], "ambiguous")
        spark.body.refresh_power.return_value = None
        self.assertIn('"charging": null', spark._body_context())

    def setUp(self):
        # Synthetic constant-level PCM exercises the energy fallback. Tests
        # of spectral VAD provide explicit speech/noise classifications.
        detector = patch("spark.ear._speech_detector", return_value=None)
        self.detector = detector.start()
        self.addCleanup(detector.stop)

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

    def test_due_idle_action_waits_for_speech_but_tap_interrupts(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = "spark"
        rec.finish.return_value = "spark"
        idle = Mock(return_value=True)
        frames = [pcm(5000)]*20 + [pcm(300)]*50
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"], idle_check=idle)
        self.assertEqual(result.text, "spark")
        idle.assert_not_called()
        self.assertFalse(listen_for_wake(iter([pcm(300)]*60), rec, CFG, ["spark"], idle_check=idle))
        idle.assert_called_once()
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"], tap_check=lambda: True))

    def test_false_voiced_noise_endpoints_and_empty_server_reply_skips_vosk(self):
        vad = Mock()
        vad.is_speech.return_value = True  # observed steady-noise VAD failure
        self.detector.return_value = vad
        room, speech, spike = pcm(450), pcm(3000), pcm(2200)
        tail = [room]*8 + [spike] + [room]*9
        frames = [room]*20 + [speech]*25 + tail*8
        mic = Mock(noise_floor=450, prefix_frames=0)
        mic.frames = lambda: iter(frames)
        recording = record_utterance(mic, CFG)
        self.assertIn(speech*25, recording)
        self.assertLess(len(recording)/32000, 1.5)
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = "park come here"
        rec.finish.return_value = "park come here"
        verify = Mock(return_value="Spark, come here.")
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 450, verify_wake=verify)
        self.assertEqual(result.text, verify.return_value)
        self.assertLess(len(verify.call_args.args[0])/32000, 1.5)
        verify.reset_mock()
        self.assertFalse(listen_for_wake(iter(tail*20), rec, CFG, ["spark"],
                                        noise_floor=lambda: 450, verify_wake=verify))
        verify.assert_not_called()

        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "asr": {"server_url": "test"}}
        spark.whisper = Mock(last_source="server")
        spark.whisper.transcribe_pcm.return_value = ""
        rec.reset_mock()
        with patch("spark.ear.record_utterance", return_value=recording):
            text, _ = spark._listen_command(mic, rec)
        self.assertEqual(text, "")
        rec.feed.assert_not_called()
        rec.finish.assert_not_called()

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

    def test_paused_okay_does_not_discard_followup_request(self):
        spark = Spark.__new__(Spark)
        spark.cfg = CFG
        spark.whisper = None
        rec, mic = Mock(), Mock()
        rec.finish.side_effect = ["Okay.", "Set a timer"]
        with patch("spark.ear.record_utterance", side_effect=[pcm(3000)*5, pcm(3000)*5]) as capture:
            text, _ = spark._listen_command(mic, rec, timeout_s=12, followup=True)
        self.assertEqual(text, "Set a timer")
        self.assertEqual(capture.call_count, 2)

    def test_constrained_keyword_rejects_a_long_forced_spark(self):
        # A constrained decoder can force unrelated loud chatter to "spark";
        # only a short wake-sized utterance may authorize locally.
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.return_value = "spark"
        speech, room = pcm(5000), pcm(1000)
        frames = [room]*5 + [speech]*90 + [room]*40
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                         noise_floor=lambda: 1000))

    def test_loud_vosk_garble_needs_independent_name_verification(self):
        # 'Spark!' can decode as 'bark', but TV saying 'bar' can too.
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.return_value = "bark"
        speech, room = pcm(5000), pcm(1000)
        frames = [room]*5 + [speech]*15 + [room]*40
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000))
        verify = Mock(return_value="Spark.")
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 1000, verify_wake=verify)
        self.assertEqual(result.text, "Spark.")
        # A quiet lookalike cannot even request the remote verifier.
        verify.reset_mock()
        self.assertFalse(listen_for_wake(iter([room]*5 + [pcm(2000)]*15 + [room]*40),
                                         rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                         verify_wake=verify))
        verify.assert_not_called()

    def test_park_in_background_conversation_needs_actual_name_verification(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        verify = Mock(return_value="A park.")
        room, speech = pcm(1000), pcm(5000)
        frames = [room]*5 + [speech]*20 + [room]*40
        rec.finish.return_value = "the park"
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000, verify_wake=verify))
        self.assertIn("the park", rec.begin.call_args.args[0])  # negative decoy, not a wake
        rec.finish.return_value = "park"
        for transcript in ("Park.", "Parks."):
            verify.return_value = transcript
            self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                            noise_floor=lambda: 1000, verify_wake=verify))
        rec.finish.return_value = "bar"
        verify.return_value = "Can you get a PR going?"
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000, verify_wake=verify))
        rec.finish.return_value = "a bar"
        verify.return_value = "Spark."
        verify.reset_mock()
        self.assertFalse(listen_for_wake(iter([room]*5 + [pcm(2000)]*20 + [room]*40),
                                         rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                         verify_wake=verify))
        verify.assert_not_called()
        rec.finish.return_value = "bar"
        self.assertEqual(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                         noise_floor=lambda: 1000, verify_wake=verify).text,
                         "Spark.")

    def test_long_exact_name_can_still_use_normal_volume_server_verification(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.return_value = "spark set a timer"
        verify = Mock(return_value="Spark, set a timer.")
        room, speech = pcm(1000), pcm(2000)  # below weak-word loudness gate
        result = listen_for_wake(iter([room]*5 + [speech]*75 + [room]*40),
                                 rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                 verify_wake=verify)
        self.assertEqual(result.text, verify.return_value)

    def test_rejected_segment_does_not_block_next_verified_name(self):
        rec = Mock()
        calls = [0]
        def finalized(_):
            calls[0] += 1
            return {15: "noise", 30: "bark"}.get(calls[0])
        rec.feed.side_effect = finalized
        rec.partial.return_value = ""
        verify = Mock(side_effect=["", "Spark."])
        room, speech = pcm(1000), pcm(5000)
        result = listen_for_wake(iter([room]*5 + [speech]*40 + [room]*40),
                                 rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                 verify_wake=verify)
        self.assertEqual(result.text, "Spark.")
        self.assertEqual(verify.call_count, 2)

    def test_muffled_name_dropped_by_whisper_wakes_via_vosk_family(self):
        # 17:35 log: the endpoint split the utterance - segment A finalized
        # 'barks [unk]' (below weak-verify loudness), segment B carried the
        # command and verified to 'Ten seconds.' with no name at all.
        from spark.ear import listen_for_wake
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.side_effect = ["barks [unk]", "", ""]
        speech, room = pcm(1500), pcm(400)
        frames = ([room]*5 + [speech]*40 + [room]*25 +   # A: the name alone
                  [speech]*15 + [room]*40)                # B: the command
        verify = Mock(return_value="Ten seconds.")
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 380, verify_wake=verify)
        self.assertTrue(result)
        self.assertEqual(result.text, "Ten seconds.")
        # (the 2.5s decay of a stale family hint is real-time behavior;
        # the frame loop consults no per-frame clock, so it is not
        # unit-testable here without deeper surgery)

    def test_loud_parakeet_bart_still_wakes(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.return_value = "barkley hear me"
        speech, room = pcm(5000), pcm(1000)
        frames = [room]*5 + [speech]*25 + [room]*40
        verify = Mock(return_value="Bart, can you hear me?")
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 1000, verify_wake=verify)
        self.assertTrue(result)
        self.assertEqual(result.text, verify.return_value)
        rec.finish.return_value = "okay bark"
        verify.return_value = "Hey Bart, can you hear me?"
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 1000, verify_wake=verify)
        self.assertEqual(result.text, verify.return_value)

    def test_noisy_room_wake_endpoint_verifies_soundalike_and_keeps_full_command(self):
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = "mark how much battery"
        rec.finish.return_value = "mark how much battery do you have"
        speech, room = pcm(3000), pcm(1000)
        frames = [room]*20 + [speech]*25 + [room]*60
        verify = Mock(return_value="Spark, how much battery do you have?")
        result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                 noise_floor=lambda: 1000, verify_wake=verify)
        self.assertEqual(result.text, verify.return_value)
        self.assertIn(speech*25, verify.call_args.args[0])
        self.assertLess(len(verify.call_args.args[0])/32000, 1.5)
        verify.return_value = "Mark, how much battery do you have?"
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000, verify_wake=verify))
        rec.reset_mock()
        self.assertFalse(listen_for_wake(iter([room]*100), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000))
        rec.feed.assert_not_called()  # steady fan noise no longer arms Vosk

    def test_wake_verification_preserves_following_audio_and_fails_closed(self):
        spark = Spark.__new__(Spark)
        spark.body = Mock(sleeping=False)
        spark.cfg = {**CFG, "asr": {"server_url": "test"}}
        spark.whisper = Mock()
        spark.whisper.transcribe_wake_pcm.return_value = "Spark."
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        mic = Mock(noise_floor=1000)
        def check(*args, **kwargs):
            return kwargs["verify_wake"](pcm(3000)*20)
        with patch("spark.ear.listen_for_wake", side_effect=check):
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ["spark"]), "Spark.")
            self.assertEqual(mic.retain.call_args.args, (3,))
            spark.whisper.transcribe_wake_pcm.side_effect = TimeoutError("offline")
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ["spark"]), "")
            self.assertEqual(mic.retain.call_args.args, (1,))

    def test_spectral_vad_ends_over_loud_noise_and_recovers_missing_wake_name(self):
        speech, fan = pcm(3000), pcm(2400)
        vad = Mock()
        vad.is_speech.side_effect = lambda frame, rate: frame == speech
        self.detector.return_value = vad
        frames = [fan]*20 + [speech]*25 + [fan]*100
        source = Mock(noise_floor=1000, prefix_frames=0)
        source.frames = lambda: iter(frames)
        recorded = record_utterance(source, CFG)
        self.assertIn(speech*25, recorded)
        self.assertLess(len(recorded)/32000, 1.5)
        source.frames = lambda: iter([fan]*200)
        self.assertEqual(record_utterance(source, CFG), b"")
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = "or how much battery"
        verify = Mock(return_value="Spark, how much battery do you have?")
        for local_text in ("or how much battery do you have", ""):
            rec.finish.return_value = local_text
            result = listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                     noise_floor=lambda: 1000, verify_wake=verify)
            self.assertEqual(result.text, verify.return_value)
        verify.reset_mock()
        self.assertFalse(listen_for_wake(iter([fan]*200), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000, verify_wake=verify))
        verify.assert_not_called()
        verify.return_value = "How much battery do you have?"
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                        noise_floor=lambda: 1000, verify_wake=verify))

    def test_mains_filter_removes_hum_preserves_speech_and_is_frame_continuous(self):
        def tone(hz):
            return struct.pack("<16000h", *(int(3000*math.sin(2*math.pi*hz*i/16000)) for i in range(16000)))
        def rms(data):
            samples = struct.unpack("<%dh" % (len(data)//2), data)
            return math.sqrt(sum(x*x for x in samples)/len(samples))
        hum, speech = tone(60), tone(1000)
        filtered = SpeechHighPass(16000).process(hum)
        self.assertLess(rms(filtered[3200:]), rms(hum)*0.2)
        self.assertGreater(rms(SpeechHighPass(16000).process(speech)[3200:]), rms(speech)*0.95)
        streaming = SpeechHighPass(16000)
        self.assertEqual(filtered, b"".join(streaming.process(hum[i:i+640]) for i in range(0,len(hum),640)))

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
        spark.body = Mock(sleeping=False, has={k: True for k in ("helper", "touch", "tts", "sound")})
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

    def test_short_signoff_closes_followup_without_reply_or_another_window(self):
        from spark.__main__ import _followup_done
        for text in ("Thanks!", "OK, thanks.", "Got it", "All set, Spark", "Thank you"):
            self.assertTrue(_followup_done(text), text)
        for text in ("thanks, set a timer", "got it, and what's the weather?",
                     "all set for tomorrow", "okay", "ok"):
            self.assertFalse(_followup_done(text), text)

        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "conversation": {"follow_up_window_s": 12}}
        spark.body = Mock(sleeping=False, has={k: True for k in ("helper", "touch", "tts", "sound")})
        spark.body.speaking_recently.return_value = False
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        spark._wait_for_wake = Mock(side_effect=[WakeResult("spark"), StopIteration])
        spark._listen_command = Mock(side_effect=[("what time is it", b""), ("OK, thanks.", b"")])
        spark.converse = Mock()
        mic = Mock()
        mic.__enter__ = Mock(return_value=mic)
        mic.__exit__ = Mock(return_value=False)
        with patch("spark.asr.Recognizer"), patch("spark.ear.MicStream", return_value=mic), \
                patch("spark.__main__.threading.Thread"), patch("spark.__main__.sd_notify"):
            with self.assertRaises(StopIteration):
                spark.voice_loop()
        self.assertEqual([c.kwargs["timeout_s"] for c in spark._listen_command.call_args_list], [6, 12])
        spark.converse.assert_called_once_with("what time is it")
        self.assertEqual(spark._wait_for_wake.call_count, 2)  # name needed again
        spark.body.speak.assert_not_called()

    def test_sleep_skips_followup_and_idle_until_name_or_tap(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "conversation": {"follow_up_window_s": 8}}
        spark.body = Body({}, hw=False)
        spark.body.has = {k: True for k in ("helper", "touch", "tts", "sound")}
        spark.body.eyes = Mock()
        spark.body.led_color = Mock()
        spark.body.dock_probe = Mock()
        spark.body.flush_sfx = Mock()
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        spark._listen_command = Mock(return_value=("go to sleep", b""))
        spark.converse = Mock(side_effect=lambda text: spark.body.sleep_pose())
        calls = []
        def wake(mic, rec, words, idle_check):
            calls.append(spark.body.sleeping)
            if len(calls) == 2:
                self.assertFalse(idle_check())
                raise StopIteration
            return WakeResult("spark")
        spark._wait_for_wake = wake
        mic = Mock()
        mic.__enter__ = Mock(return_value=mic)
        mic.__exit__ = Mock(return_value=False)
        with patch("spark.asr.Recognizer"), patch("spark.ear.MicStream", return_value=mic), \
                patch("spark.__main__.threading.Thread"), patch("spark.__main__.sd_notify"):
            with self.assertRaises(StopIteration):
                spark.voice_loop()
        self.assertEqual(calls, [False, True])
        spark._listen_command.assert_called_once()
        self.assertTrue(spark.body.actuators_held())
        spark.body.wake_up()
        self.assertFalse(spark.body.sleeping)

    def test_okay_spark_and_bounded_spoken_replies(self):
        from spark.ear import has_wake_name, strip_wake_prefix
        from spark.brain import spoken_sentences
        self.assertTrue(has_wake_name("Okay Spark, hi", ["spark"]))
        self.assertEqual(strip_wake_prefix("Okay Spark, hi"), "hi")
        self.assertFalse(has_wake_name("Sparkling water", ["spark"]))
        closed = []
        def deltas():
            try:
                for sentence in ("Hello. ", "How are you? ", "More unwanted chatter. "):
                    yield sentence
            finally:
                closed.append(True)
        self.assertEqual(list(spoken_sentences(deltas())), ["Hello.", "How are you?"])
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
