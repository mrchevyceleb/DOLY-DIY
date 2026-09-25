"""Brain-triggered web tools: marker parsing, page fetch, tool loop."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spark import search as websearch
from spark.__main__ import Spark


class ParseToolCallTests(unittest.TestCase):
    def test_search_marker(self):
        self.assertEqual(websearch.parse_tool_call("SEARCH: lakers score last night"),
                         ("search", "lakers score last night"))

    def test_search_marker_strips_trailing_punctuation(self):
        self.assertEqual(websearch.parse_tool_call("Search: who won the world series."),
                         ("search", "who won the world series"))

    def test_read_marker_keeps_url(self):
        self.assertEqual(websearch.parse_tool_call("READ: https://example.com/a?b=1"),
                         ("read", "https://example.com/a?b=1"))

    def test_read_requires_http_url(self):
        self.assertIsNone(websearch.parse_tool_call("READ: /etc/passwd"))
        self.assertIsNone(websearch.parse_tool_call("read: ftp://x"))

    def test_ordinary_speech_is_not_a_tool(self):
        for line in ("Search engines are neat.", "I read that book last week.",
                     "Let me search my memory for it.", "", "Reading: my favorite part."):
            self.assertIsNone(websearch.parse_tool_call(line), line)

    def test_quoted_argument(self):
        self.assertEqual(websearch.parse_tool_call('SEARCH: "nvidia stock price"'),
                         ("search", "nvidia stock price"))


class ReadPageTests(unittest.TestCase):
    PAGE = (b"<html><head><title> Cool Gadget </title>"
            b"<style>body{color:red}</style></head>"
            b"<body><nav>Home About</nav>"
            b"<p>The gadget ships in October for $499.</p>"
            b"<script>alert('evil')</script>"
            b"<p>It weighs two pounds.</p></body></html>")

    def _fetch(self, data=PAGE, ctype="text/html; charset=utf-8"):
        resp = Mock()
        resp.headers = {"Content-Type": ctype}
        resp.read.return_value = data
        resp.__enter__ = Mock(return_value=resp)
        resp.__exit__ = Mock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            return websearch.read_page("https://example.com/review")

    def test_extracts_title_and_text(self):
        page = self._fetch()
        self.assertEqual(page["title"], "Cool Gadget")
        self.assertIn("ships in October for $499", page["text"])
        self.assertIn("two pounds", page["text"])

    def test_strips_unreadable_blocks(self):
        page = self._fetch()
        self.assertNotIn("alert", page["text"])
        self.assertNotIn("Home About", page["text"])
        self.assertNotIn("color:red", page["text"])

    def test_caps_length(self):
        page = self._fetch(data=b"<p>" + b"word " * 5000 + b"</p>")
        self.assertLessEqual(len(page["text"]), 3502)  # cap + ellipsis

    def test_rejects_binary_content(self):
        self.assertIsNone(self._fetch(ctype="application/pdf"))

    def test_rejects_non_http(self):
        self.assertIsNone(websearch.read_page("file:///etc/passwd"))
        self.assertIsNone(websearch.read_page(""))

    def test_page_block_render(self):
        block = websearch.page_block("https://x.io", {"title": "T", "text": "hello"})
        self.assertIn("hello", block)
        self.assertIn("T", block)


class ToolLoopTests(unittest.TestCase):
    """The brain's SEARCH line executes the tool and re-asks with results."""

    def _spark(self):
        spark = Spark.__new__(Spark)
        spark.cfg = {"prompt": "You are Spark.", "moods": False,
                     "web": {"enabled": True, "max_hops": 2,
                             "page_max_chars": 100, "page_timeout_s": 6}}
        spark.body = Mock()
        spark.body.mood = "happy"
        spark.body.speak_stream.side_effect = lambda gen: list(gen)  # consume like the real pipeline
        spark._body_context = lambda: ""
        spark.memory = Mock()
        spark.memory.messages.return_value = [{"role": "user", "content": "q"}]
        spark.brain = Mock()
        spark.brain_online = True
        return spark

    def test_search_tool_round_trip(self):
        spark = self._spark()
        spark.brain.chat_stream.side_effect = [
            iter(["SEARCH: who won the game."]),   # first ask: demand the web
            iter(["The Chiefs won.", " It was close."]),  # re-ask with results
        ]
        with patch.object(websearch, "web_search",
                          return_value=[{"title": "t", "snippet": "s",
                                         "url": "https://x"}]) as ws:
            spark._llm_reply("who won the game?")
            ws.assert_called_once()
        # user recorded once, tool line never spoken, final reply remembered
        self.assertEqual(spark.memory.add.call_count, 2)
        self.assertEqual(spark.memory.add.call_args_list[0][0], ("user", "who won the game?"))
        self.assertEqual(spark.memory.add.call_args_list[1][0],
                         ("assistant", "The Chiefs won. It was close."))
        self.assertEqual(spark.body.speak.call_args[0][0], "Let me look that up.")

    def test_read_tool_after_search(self):
        spark = self._spark()
        spark.brain.chat_stream.side_effect = [
            iter(["SEARCH: mars weather."]),
            iter(["READ: https://nasa.gov/mars."]),
            iter(["It is minus eighty."]),
        ]
        with patch.object(websearch, "web_search", return_value=[
                {"title": "Mars", "snippet": "s", "url": "https://nasa.gov/mars"}]), \
             patch.object(websearch, "read_page",
                          return_value={"title": "Mars Weather", "text": "-80F"}):
            spark._llm_reply("what is the weather on mars?")
        self.assertEqual(spark.memory.add.call_args_list[-1][0],
                         ("assistant", "It is minus eighty."))

    def test_failed_search_degrades_gracefully(self):
        spark = self._spark()
        spark.brain.chat_stream.side_effect = [
            iter(["SEARCH: stock price."]),
            iter(["I could not check that right now."]),
        ]
        with patch.object(websearch, "web_search", return_value=[]):
            spark._llm_reply("what is the stock price?")
        self.assertEqual(spark.memory.add.call_args_list[-1][0],
                         ("assistant", "I could not check that right now."))

    def test_web_disabled_disables_tools(self):
        spark = self._spark()
        spark.cfg["web"]["enabled"] = False
        spark.brain.chat_stream.return_value = iter(["SEARCH: anything."])
        spark._llm_reply("hello?")
        # no tool executed: the marker line is spoken as ordinary speech
        self.assertEqual(spark.memory.add.call_count, 2)
        self.assertEqual(spark.memory.add.call_args_list[1][0],
                         ("assistant", "SEARCH: anything."))


if __name__ == "__main__":
    unittest.main()
