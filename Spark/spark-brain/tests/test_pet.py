import unittest

from spark import commands as cmds
from spark.pet import _PRAISE_RE, classify_trick, parse_teach, rps_result


class PetTests(unittest.TestCase):
    def test_teach_with_and_without_comma(self):
        raw = "Spark, when I say bang, play dead."
        self.assertEqual(parse_teach(raw, cmds.normalize(raw)), ("bang", "play dead"))
        self.assertEqual(parse_teach("", "when i say marco say polo"), ("marco", "say polo"))

    def test_trick_classes(self):
        self.assertEqual(classify_trick("bark like a dog"), ("animal", "dog"))
        self.assertEqual(classify_trick("say 'Polo'"), ("say", "polo"))
        self.assertEqual(classify_trick("spin around"), ("spin", None))
        self.assertIsNone(classify_trick("do my taxes"))

    def test_praise_is_whole_utterance_only(self):
        for said in ("Good girl, Spark!", "You're the best.", "Thank you so much!", "I love you"):
            self.assertTrue(_PRAISE_RE.match(cmds.normalize(said)), said)
        self.assertFalse(_PRAISE_RE.match(cmds.normalize("Thank you, what time is it?")))

    def test_rps(self):
        self.assertEqual(rps_result("paper", "rock"), "me")
        self.assertEqual(rps_result("rock", "paper"), "you")
        self.assertEqual(rps_result("scissors", "scissors"), "tie")


if __name__ == "__main__":
    unittest.main()
