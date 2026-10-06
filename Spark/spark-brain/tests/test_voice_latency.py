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
    def test_neural_wake_keeps_consumed_audio_and_command_after_room_speech(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, 'asr': {'server_url': 'test'}, 'wake': {'words': ['hey spark']}}
        spark.body = Mock(sleeping=False)
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        spark.whisper = Mock(last_source='server')
        spark.whisper.transcribe_pcm.return_value = 'Room chatter. Hey Spark, dim the lights.'
        spark._neural_wake = Mock(shadow=False, phrase='hey spark', last_score=.99)
        spark._neural_wake.feed.side_effect = [False]*19+[True]
        quiet_greeting = pcm(200)*20  # below the old absolute onset gate
        mic = Mock(noise_floor=40)
        mic.frames = lambda: iter([pcm(200)]*20)
        rec = Mock(feed=Mock(return_value=None),partial=Mock(return_value=''))
        with patch('spark.ear._speech_detector',return_value=None):
            wake = spark._wait_for_wake(mic,rec,['hey spark'])
        self.assertTrue(wake.neural)
        self.assertEqual(wake.prefix_pcm,quiet_greeting)
        spark.whisper.transcribe_wake_pcm.assert_not_called()
        def capture(audio, *args, **kwargs):
            prefix = [next(audio.frames()) for _ in range(20)]
            self.assertEqual(b''.join(prefix),quiet_greeting)
            return quiet_greeting+pcm(300)*20
        with patch('spark.ear.record_utterance',side_effect=capture):
            text,_ = spark._listen_command(mic,rec,wake)
        self.assertEqual(text,'dim the lights.')
        # An earlier sentence and pause in the rolling audio must not end
        # capture before the greeting and its request are replayed.
        previous = pcm(2000)*10+pcm(0)*30
        greeting = pcm(2000)*20
        mic.frames = lambda: iter([pcm(2000)]*15+[pcm(0)]*25)
        audio = CommandAudio(mic,previous+greeting)
        with patch('spark.ear._speech_detector',return_value=None):
            captured = record_utterance(audio,CFG,endpoint_after_frames=audio.prefix_frames)
        self.assertGreater(len(captured),len(previous+greeting))
        spark.whisper.transcribe_pcm.return_value = 'Hey Spark, are you there?'
        with patch('spark.ear.record_utterance',return_value=pcm(2000)*30):
            text,_ = spark._listen_command(mic,rec,wake)
        self.assertEqual(text,'are you there?')
        spark.whisper.confirm_wake_pcm.assert_not_called()
        spark.cfg['asr']['wake_server_url'] = 'test'
        spark.whisper.transcribe_pcm.return_value = 'Dim the lights.'
        spark.whisper.confirm_wake_pcm.return_value = 'Hey Spark.'
        with patch('spark.ear.record_utterance',return_value=pcm(2000)*30):
            text,_ = spark._listen_command(mic,rec,wake)
        self.assertEqual(text,'Dim the lights.')
        # High classifier confidence without an addressed greeting must not
        # execute an incidental room command or lead to repeated miss prompts.
        spark.cfg['asr']['wake_server_url'] = 'test'
        spark.whisper.transcribe_pcm.return_value = 'Dim the lights.'
        spark.whisper.confirm_wake_pcm.return_value = 'Dim the lights.'
        with patch('spark.ear.record_utterance',return_value=pcm(2000)*30):
            text,_ = spark._listen_command(mic,rec,wake)
        self.assertEqual(text,'')
        self.assertTrue(wake.neural_unconfirmed)

    def test_neural_inference_failure_and_shadow_keep_legacy_listener(self):
        spark = Spark.__new__(Spark)
        spark.cfg = CFG
        spark.body = Mock(sleeping=False)
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        mic = Mock(noise_floor=40)
        mic.frames = lambda: iter([pcm(200)]*10)
        legacy = WakeResult('hey spark')
        def fallback(frames,*args,**kwargs):
            list(frames)
            return legacy
        for shadow,failure in ((True,False),(True,True),(False,False)):
            neural = Mock(shadow=shadow,last_score=.99)
            neural.feed.side_effect = RuntimeError('bad model') if failure else None
            neural.feed.return_value = shadow
            spark._neural_wake = neural
            with patch('spark.ear.listen_for_wake',side_effect=fallback):
                self.assertIs(spark._wait_for_wake(mic,Mock(),['hey spark']),legacy)
            if failure:self.assertIsNone(spark._neural_wake)

    def test_prompt_history_keeps_its_prefix_and_prunes_in_bounded_batches(self):
        import tempfile
        from spark.memory import Memory
        with tempfile.TemporaryDirectory() as state:
            memory = Memory({'brain': {'history_turns': 2, 'history_context_batch_turns': 2},
                             'state_dir': state})
            for i in range(4):
                prefix = memory.messages('Spark')
                memory.add('user', f'Question {i}')
                memory.add('assistant', f'Answer {i}')
                self.assertEqual(memory.messages('Spark')[:len(prefix)], prefix)
            memory.add('user', 'Question 4')
            memory.add('assistant', 'Answer 4')
            sent = memory.messages('Spark')
            self.assertEqual(sent[1]['content'], 'Question 2')
            self.assertEqual(sent[-1]['content'], 'Answer 4')
            self.assertEqual(len(memory.history), 4)
            self.assertLessEqual(len(sent), 9)
            loaded = Memory({'brain': {'history_turns': 2}, 'state_dir': state})
            self.assertEqual(loaded.messages('Spark')[1:], list(memory.history))
            memory.clear()
            self.assertEqual(memory.messages('Spark'), [{'role': 'system', 'content': 'Spark'}])
            for i in range(2):
                memory.add('user', f'Question {i}')
                memory.add('assistant', f'Answer {i}')
            memory.add('user', 'Long question ' + 'x' * 6000)
            self.assertEqual(memory.messages('Spark')[1]['role'], 'user')

    def test_availability_question_is_fast_but_does_not_swallow_more_instructions(self):
        from spark.router import Router
        router = Router.__new__(Router)
        router.body, router.brain = Mock(), Mock()
        router.cfg = {'govee': {'enabled': False}}
        router.govee = None
        for text in ('Are you okay now?', 'Can you hear me?', 'Are you still there?'):
            self.assertTrue(router.handle(text))
            router.body.speak.assert_called_with("I'm here, Matt. I'm listening.")
        with patch('spark.router.actions.wanted', return_value=False), \
                patch.object(router, '_motion_followup', return_value=False):
            for text in ('Are you okay with that plan?', 'Can you hear me and dim the lights?'):
                router.body.reset_mock()
                self.assertFalse(router.handle(text))
                router.body.speak.assert_not_called()
        router.brain.chat.assert_not_called()
        # The same voice stays available even after ordinary LRU eviction.
        import importlib.util
        import wave
        module_path = Path(__file__).resolve().parents[1] / 'deploy/moria/qwen_voice_server.py'
        spec = importlib.util.spec_from_file_location('voice_server_latency_check', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        voice = module.Voice.__new__(module.Voice)
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            wav.writeframes(pcm(1000))
        voice.fixed_cache = {"I'm here, Matt. I'm listening.": output.getvalue()}
        voice.cache = {}
        self.assertEqual(voice.generate("I'm here, Matt. I'm listening.")[1], 0)
        self.assertEqual(b''.join(voice.stream("I'm here, Matt. I'm listening.")), pcm(1000))


    def test_exact_greeting_needs_two_matching_addressed_commands_for_name_recovery(self):
        from spark.ear import DeferredWake, corroborated_local_wake
        words = ['hey spark']
        good = ("Face Park, come here.", "Thanks, Park. Come here.")
        self.assertEqual(corroborated_local_wake('hey spark [unk]', *good, words), 'hey spark, come here.')
        for local, first, second in (
            ('spark', *good), ('a spark', *good),
            ('hey spark', good[0], 'Park, turn around.'),
            ('hey spark', 'Park the car.', 'Park the car.'),
            ('hey spark', 'Central Park, come here.', 'Central Park, come here.'),
            ('hey spark', good[0], ''),
            ('hey spark extra', *good), ('okay hey spark', *good),
        ):
            self.assertIsNone(corroborated_local_wake(local, first, second, words))
        cfg = {'audio': dict(CFG['audio'], adaptive_sensitivity=True)}
        frames = [pcm(40)]*20 + [pcm(180)]*30 + [pcm(40)]*40
        rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value='hey spark'),
                   finish=Mock(return_value='hey spark'))
        exact = Mock(side_effect=[DeferredWake('hey spark, come here.'), AssertionError('confirmation decoded twice')])
        with patch('spark.ear._speech_detector', return_value=Mock(is_speech=lambda *args: True)):
            result = listen_for_wake(iter(frames), rec, cfg, words, noise_floor=lambda:40,
                                     verify_wake=Mock(), verify_exact=exact)
        self.assertTrue(result)
        self.assertEqual([call.args[2] for call in exact.call_args_list], [False])

    def test_soft_speech_is_captured_without_treating_room_noise_as_speech(self):
        from spark.ear import _rms
        audio = dict(CFG['audio'], start_rms=250, stop_rms=125)
        vad = Mock(is_speech=lambda frame, rate: _rms(frame) >= 200)
        for adaptive, heard in ((False, False), (True, True)):
            cfg = {'audio': dict(audio, adaptive_sensitivity=adaptive)}
            source = Mock(noise_floor=40, prefix_frames=0)
            source.frames = lambda: iter([pcm(40)]*20 + [pcm(n) for n in [110,140,120]*10] + [pcm(40)]*40)
            with patch('spark.ear._speech_detector', return_value=vad):
                self.assertEqual(bool(record_utterance(source, cfg)), heard)
        source = Mock(noise_floor=140, prefix_frames=0)
        source.frames = lambda: iter([pcm(140)]*100)
        with patch('spark.ear._speech_detector', return_value=vad):
            self.assertFalse(record_utterance(source, cfg))
        source.noise_floor = 40  # fan baseline rises after the previous estimate
        source.frames = lambda: iter([pcm(40)]*20 + [pcm(100)]*100)
        with patch('spark.ear._speech_detector', return_value=Mock(is_speech=lambda *args: True)):
            self.assertFalse(record_utterance(source, cfg))

    def test_word_timing_survives_finalized_name_and_resets_with_next_decode(self):
        import json
        from spark.asr import Recognizer
        rec = Recognizer.__new__(Recognizer)
        native = Mock()
        native.AcceptWaveform.return_value = True
        native.Result.return_value = json.dumps({"text": "hey spark", "result": [
            {"word": "hey", "end": .3}, {"word": "spark", "end": .8}]})
        native.FinalResult.return_value = json.dumps({"text": "weather", "result": [
            {"word": "weather", "end": 2.0}]})
        rec._kaldi_cls, rec.model, rec.sample_rate = Mock(return_value=native), None, 16000
        rec.begin()
        rec.feed(pcm(1800))
        self.assertEqual(rec.finish(), "hey spark weather")
        self.assertEqual([w["word"] for w in rec.last_words], ["hey", "spark", "weather"])
        rec.begin()
        self.assertEqual(rec.last_words, [])

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
        with patch("spark.ear.record_utterance", return_value=b""):
            text, _ = spark._listen_command(Mock(), Mock(), WakeResult("spark stop"))
        self.assertEqual(text, "stop")
        # The verified clip was cut mid-sentence ('Park. What's'): the rest
        # queued in the mic during the check completes the command.
        spark.whisper.transcribe_pcm.side_effect = None
        spark.whisper.transcribe_pcm.return_value = "the weather today?"
        with patch("spark.ear.record_utterance", return_value=pcm(3000)*30):
            text, _ = spark._listen_command(Mock(noise_floor=500), Mock(),
                                            WakeResult("Park. What's", command="What's"))
        self.assertEqual(text, "What's the weather today?")
        with patch("spark.ear.record_utterance", side_effect=AssertionError):
            text, _ = spark._listen_command(Mock(), Mock(), WakeResult("Spark, stop.", partial=True))
        self.assertEqual(text, "stop.")

    def test_partial_wake_punctuation_keeps_the_rest_of_the_light_command(self):
        speech, room = pcm(5000), pcm(500)
        frames = iter([room]*5 + [speech]*60 + [room]*40)
        rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value="hey spark"))
        wake = listen_for_wake(frames, rec, CFG, ["hey spark"],
                               noise_floor=lambda: 500,
                               verify_wake=Mock(return_value="Hey Spark, dimble."))
        self.assertTrue(wake.partial)
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, "asr": {"server_url": "test"}}
        spark.whisper = Mock()
        spark.whisper.transcribe_pcm.return_value = "Hey Spark, dim the lights to thirty."
        mic = Mock(noise_floor=500)
        mic.frames = lambda: frames
        text, _ = spark._listen_command(mic, rec, wake)
        self.assertEqual(text, "dim the lights to thirty.")
        recording = spark.whisper.transcribe_pcm.call_args.args[0]
        self.assertTrue(recording.startswith(wake.clip_pcm))
        self.assertGreater(len(recording), len(wake.clip_pcm))
        # Silence after an early completed command adds no second ASR call.
        spark.whisper.reset_mock()
        mic.frames = lambda: iter([room]*40)
        wake.text = "Hey Spark, lights off."
        text, _ = spark._listen_command(mic, rec, wake)
        self.assertEqual(text, "lights off.")
        spark.whisper.transcribe_pcm.assert_not_called()
        # An endpointed command must not absorb unrelated room talk.
        wake.partial = False
        with patch("spark.ear.record_utterance", side_effect=AssertionError):
            text, _ = spark._listen_command(mic, rec, wake)
        self.assertEqual(text, "lights off.")

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
        # A speaking-level lookalike is checked (live misses at ~2600), but
        # only the verifier hearing her actual name may wake her.
        verify.reset_mock()
        verify.return_value = "The dog's bark."
        self.assertFalse(listen_for_wake(iter([room]*5 + [pcm(2000)]*15 + [room]*40),
                                         rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                         verify_wake=verify))
        verify.assert_called_once()
        # Parakeet drops the soft 's' ('Park, ...') on quiet speech; the
        # vocative comma separates it from talk about a park.
        verify.return_value = "Park, what's the weather today?"
        quiet = [room]*5 + [pcm(2000)]*15 + [room]*40
        result = listen_for_wake(iter(quiet), rec, CFG, ["spark"],
                                 noise_floor=lambda: 1000, verify_wake=verify)
        self.assertEqual(result.command, "what's the weather today?")
        for transcript in ("Park the car.", "Sparks had a timer.", "Hey Park.", "The dog's bark, again."):
            verify.return_value = transcript
            self.assertFalse(listen_for_wake(iter(quiet), rec, CFG, ["spark"],
                                             noise_floor=lambda: 1000, verify_wake=verify))
        rec.finish.return_value = "sparky"
        verify.return_value = "More softly."
        self.assertFalse(listen_for_wake(iter(quiet), rec, CFG, ["spark"],
                                         noise_floor=lambda: 1000, verify_wake=verify))

    def test_hey_phrase_is_required_across_wake_paths(self):
        from spark.ear import addressed_command, has_wake_name
        words = ["hey spark", "hey sparky"]
        for text in ("Spark.", "Sparky, hi.", "Okay Spark.", "Hey Park."):
            self.assertFalse(has_wake_name(text, words), text)
        self.assertTrue(has_wake_name("Hey, Spark!", words))
        self.assertIsNone(addressed_command("Weather today, Spark?", words))
        self.assertIsNone(addressed_command("Hello. Spark, weather today?", words))
        self.assertEqual(addressed_command("Weather today, hey Spark?", words), "Weather today?")
        frames = [pcm(730)]*5 + [pcm(8303)]*30 + [pcm(730)]*40
        for local, verified, accepted in (
            ("spark", None, False), ("hey spark", None, True),
            ("hey spark", "", True), ("hey spark", "Spark.", False),
            ("hey spark", "A spark.", True), ("hey sparky", "A Sparky!", True),
            ("spark", "A spark.", False), ("bark", "A spark.", False),
            ("hey spark", "A spark started the fire.", False),
            ("hey spark", "Park, stop.", False), ("hey spark", "Hey Spark.", True),
            ("bark", "", False), ("bart", "Bart.", False),
            ("barkley", "Weather today.", False), ("bark", "Park, stop.", False),
            ("spark", "Spark, weather today?", False),
            ("spark", "Hello. Spark, weather today?", False),
            ("spark", "Hey Spark, weather today?", True),
            ("bark", "Hello. Hey Spark, weather today?", True),
            ("bark", "Weather today, hey Spark?", True),
        ):
            for partial in ("", local):
                with self.subTest(local=local, verified=verified, partial=partial), \
                        patch("sys.stderr", new=io.StringIO()):
                    rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value=partial),
                               finish=Mock(return_value=local))
                    verify = None if verified is None else Mock(return_value=verified)
                    result = listen_for_wake(iter(frames), rec, CFG, words,
                                             noise_floor=lambda: 730, verify_wake=verify)
                    self.assertEqual(bool(result), accepted)
        # A partial 'A spark' must wait for the rest of background speech.
        rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value="hey spark"),
                   finish=Mock(return_value="hey spark"))
        verify = Mock(side_effect=["A spark.", "A spark started the fire."])
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, words,
                                        noise_floor=lambda: 730, verify_wake=verify))
        self.assertEqual(verify.call_count, 2)

    def test_short_loud_name_garble_opens_listening_when_verifier_is_silent(self):
        from spark.ear import RoomTalk
        rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value=""),
                   finish=Mock(return_value="bark"))
        verify = Mock(return_value="")
        busy = RoomTalk()
        for _ in range(3):
            busy.note()
        def listen(level=8303, duration=30, allow_weak=True, wake_words=("spark",)):
            frames = [pcm(730)]*5 + [pcm(level)]*duration + [pcm(730)]*40
            return listen_for_wake(iter(frames), rec, CFG, wake_words, room=busy,
                                   noise_floor=lambda: 730, verify_wake=verify,
                                   allow_weak=allow_weak)
        result = listen()
        self.assertEqual(result.text, "Spark")
        self.assertFalse(result.command)  # only opens listening
        self.assertFalse(result.prefix_pcm)
        verify.assert_called_once_with(result.clip_pcm, keep=True)
        self.assertFalse(listen(level=2500))
        self.assertFalse(listen(duration=80))
        self.assertFalse(listen(allow_weak=False))
        self.assertFalse(listen(wake_words=("nova",)))
        self.assertFalse(listen(wake_words=()))
        for text in ("bar", "park", "stark", "bark [unk]"):
            rec.finish.return_value = text
            self.assertFalse(listen(), text)
        rec.finish.return_value = "bark"
        for text in ("The dog's bark.", "Yeah.", "Hey Park."):
            verify.return_value = text
            self.assertFalse(listen(), text)

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
        # Live miss: a real 'Spark' decoded as 'bar' at 2943 was never
        # checked. Speaking-level 'bar' is verified; her name still decides.
        rec.finish.return_value = "a bar"
        verify.return_value = "Spark."
        self.assertEqual(listen_for_wake(iter([room]*5 + [pcm(2000)]*20 + [room]*40),
                                         rec, CFG, ["spark"], noise_floor=lambda: 1000,
                                         verify_wake=verify).text, "Spark.")
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

    def test_background_talk_cannot_wake_via_exact_or_family_hit(self):
        # 16:22 'sparky' -> 'Fucking background.' and, during a YouTube video,
        # 'barkley' -> 'only loses on quality.' both started a turn.
        from spark.ear import RoomTalk
        speech, room = pcm(3500), pcm(1000)
        frames = [room]*5 + [speech]*30 + [room]*40
        rec = Mock()
        rec.feed.return_value = None
        rec.partial.return_value = ""
        rec.finish.return_value = "sparky"
        verify = Mock(return_value="Fucking background.")
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                         noise_floor=lambda: 1000, verify_wake=verify))
        verify.return_value = ""  # verifier silent: the exact hit stands
        self.assertEqual(listen_for_wake(iter(frames), rec, CFG, ["spark"],
                                         noise_floor=lambda: 1000, verify_wake=verify).text,
                         "sparky")
        rec.finish.return_value = "barkley"
        verify.return_value = "only loses on quality."
        busy = RoomTalk()
        for _ in range(3):
            busy.note()
        self.assertFalse(listen_for_wake(iter(frames), rec, CFG, ["spark"], room=busy,
                                         noise_floor=lambda: 1000, verify_wake=verify))
        verify.return_value = "Spark, how tall is that?"
        self.assertTrue(listen_for_wake(iter(frames), rec, CFG, ["spark"], room=busy,
                                        noise_floor=lambda: 1000, verify_wake=verify))
        from spark.__main__ import _hushed
        for text in ("Be quiet.", "Spark, shut up!", "Shh.", "Okay, stop talking"):
            self.assertTrue(_hushed(text), text)
        self.assertFalse(_hushed("quiet the lights"))

    def test_busy_room_rechecks_dropped_wake_prefix_without_authorizing_kws_alone(self):
        from spark.ear import RoomTalk
        busy = RoomTalk()
        for _ in range(3):
            busy.note()
        rec = Mock(feed=Mock(return_value=None), partial=Mock(return_value=""),
                   finish=Mock(return_value="a spark [unk]"))
        rec.last_words = [{"word": "a", "end": .2}, {"word": "spark", "end": .75},
                          {"word": "[unk]", "end": 1.4}]
        frames = [pcm(400)]*5 + [pcm(1800)]*35 + [pcm(400)]*40
        for full, prefix, accepted in (("What's the weather?", "Hey Spark.", True),
                                       ("What's the weather?", "What's the weather?", False),
                                       ("What's the weather?", "A spark started a fire.", False),
                                       ("What's the weather?", "A spark.", False),
                                       ("Hate Spark, what's the weather?", "Hate Spark.", True),
                                       ("I hate Spark, what's the weather?", "Hate Spark.", False),
                                       ("Hate Spark is a character.", "Hate Spark.", False),
                                       ("Hate Spark, what's the weather?", "Hate Sparky.", False)):
            with self.subTest(full=full, prefix=prefix):
                verify = Mock(side_effect=[full, prefix])
                result = listen_for_wake(iter(frames), rec, CFG, ["hey spark"], room=busy,
                                         noise_floor=lambda: 380, verify_wake=verify)
                self.assertEqual(bool(result), accepted)
                if accepted:
                    self.assertEqual(result.command.casefold(), "What's the weather?".casefold())
                    self.assertIn(pcm(1800)*35, result.clip_pcm)
                self.assertEqual(len(verify.call_args.args[0]), int(16000*.87)*2)

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
            self.assertEqual(mic.retain.call_args.args, (5,))
            spark.whisper.transcribe_wake_pcm.side_effect = TimeoutError("offline")
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ["spark"]), "")
            self.assertEqual(mic.retain.call_args.args, (1,))
            spark.cfg['asr']['wake_server_url'] = 'secondary'
            spark.whisper.transcribe_wake_pcm.side_effect = None
            spark.whisper.transcribe_wake_pcm.return_value = "East Park, what's the weather?"
            spark.whisper.confirm_wake_pcm.return_value = "Hey Spark, what's the weather?"
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ['hey spark']),
                             "Hey Spark, what's the weather?")
            # Live: Vosk [unk], primary "Burke, how did you sleep?" skipped
            # the second model. No spelling hint is needed to consult it.
            spark.whisper.transcribe_wake_pcm.return_value = 'Burke, how did you sleep?'
            spark.whisper.confirm_wake_pcm.return_value = 'Hey Spark, how did you sleep?'
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ['hey spark']),
                             'Hey Spark, how did you sleep?')
            mic.save_wake_diagnostic.assert_called_with(pcm(3000)*20,
                'Burke, how did you sleep?', 'Hey Spark, how did you sleep?')
            spark.whisper.confirm_wake_pcm.return_value = 'The park is open today.'
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ['hey spark']),
                             "[wake unconfirmed]")  # vetoes even an exact local hit
            spark.whisper.confirm_wake_pcm.reset_mock()
            with patch('spark.__main__.time.monotonic', side_effect=[0, 3]):
                spark._wait_for_wake(mic, Mock(), ['hey spark'])
            spark.whisper.confirm_wake_pcm.assert_not_called()  # total check budget
        spark.whisper.transcribe_wake_pcm.return_value = 'Face Park, come here.'
        spark.whisper.confirm_wake_pcm.return_value = 'Thanks, Park. Come here.'
        def check_exact(*args, **kwargs):
            return kwargs['verify_exact'](pcm(3000)*20, 'hey spark', True)
        with patch('spark.ear.listen_for_wake', side_effect=check_exact):
            self.assertEqual(spark._wait_for_wake(mic, Mock(), ['hey spark']),
                             'hey spark, come here.')

    def test_wake_diagnostics_pairs_audio_rotates_and_expires(self):
        import json, tempfile, wave
        from spark.wake_diagnostics import WakeDiagnostics
        with tempfile.TemporaryDirectory() as folder, \
                patch('spark.wake_diagnostics.time.time', return_value=100) as clock:
            diagnostic = WakeDiagnostics(16000, 150, folder)
            diagnostic.append(pcm(300), pcm(100))
            diagnostic.append(pcm(400), pcm(200))
            for _ in range(21):
                diagnostic.save(pcm(100)+pcm(200), 'Burke.', 'Hey Spark.')
            self.assertEqual(len(list(Path(folder).glob('*.json'))), 20)
            with wave.open(str(Path(folder)/'00-raw.wav')) as wav:
                self.assertEqual(wav.readframes(wav.getnframes()), pcm(300)+pcm(400))
            diagnostic.save(pcm(999), 'unmatched', '')
            self.assertFalse(json.loads((Path(folder)/'01.json').read_text())['raw_matched'])
            self.assertFalse((Path(folder)/'01-raw.wav').exists())
            clock.return_value = 151
            diagnostic.append(pcm(500), pcm(300))
            diagnostic.save(pcm(300), 'expired', '')
            self.assertFalse(diagnostic.frames)
            self.assertEqual(diagnostic.count, 22)

    def test_name_said_to_her_mid_or_end_of_sentence_wakes_her(self):
        from spark.ear import addressed_command
        # Live on a call: both were dropped as room talk.
        self.assertEqual(addressed_command("What's the weather today, Spark?"),
                         "What's the weather today?")
        self.assertEqual(addressed_command("He's going well. Spark, what's the weather?"),
                         "what's the weather?")
        self.assertEqual(addressed_command("aren't you? Spark."), "")
        for about_her in ("You guys met have you guys met Spark?",
                          "Um and New Spark came out fast.", "I love it. Spark is great."):
            self.assertIsNone(addressed_command(about_her))

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

    def test_plural_wake_and_name_only_handoff_do_not_mute_the_next_command(self):
        from spark.ear import has_wake_name, strip_wake_prefix
        from spark.__main__ import _plausible_speech
        words = ['hey spark', 'hey sparky', 'hey sparks']
        self.assertTrue(has_wake_name('Hey Sparks.', words))
        self.assertFalse(has_wake_name('Sparks.', words))
        self.assertFalse(has_wake_name('He sparks a discussion.', words))
        self.assertEqual(strip_wake_prefix('Hey Sparks.', 'Hey Sparks.'), '')
        self.assertEqual(strip_wake_prefix('Hey Sparks, tell me a joke.'), 'tell me a joke.')
        self.assertTrue(_plausible_speech('Joke.', 3449))
        self.assertFalse(_plausible_speech('Ta.', 3449))
        spark = Spark.__new__(Spark)
        spark.cfg = {**CFG, 'conversation': {'follow_up_window_s': 0}}
        spark.body = Mock(sleeping=False, has={k: True for k in ('helper','touch','tts','sound')})
        spark.body.speaking_recently.return_value = False
        spark.talk_trigger = Mock()
        spark.talk_trigger.is_set.return_value = False
        spark._wait_for_wake = Mock(side_effect=[WakeResult('Hey Sparks.', clip_pcm=pcm(3000)*20), StopIteration])
        spark._listen_command = Mock(return_value=('Joke.', pcm(3449)*30))
        spark.converse = Mock()
        mic = Mock()
        mic.__enter__ = Mock(return_value=mic)
        mic.__exit__ = Mock(return_value=False)
        with patch('spark.asr.Recognizer'), patch('spark.ear.MicStream', return_value=mic), \
                patch('spark.__main__.threading.Thread'), patch('spark.__main__.sd_notify'):
            with self.assertRaises(StopIteration):
                spark.voice_loop()
        spark.body.wake_reaction.assert_called_once_with(audible=False)
        mic.mute.assert_not_called()
        spark.converse.assert_called_once_with('Joke.')

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
