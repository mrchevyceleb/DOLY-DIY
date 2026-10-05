"""Exact local volume requests; unrelated uses of 'up/down' stay untouched."""
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))


def intent(text):
    text = re.sub(r"^(?:(?:please|okay|ok|actually|can you|could you|would you|will you|just)\s+)+", "", text)
    text = re.sub(r"\s+please$", "", text)
    step = re.fullmatch(r"(?:turn (?:it|(?:the |your )?volume)|volume) (up|down)", text)
    if not step:
        step = re.fullmatch(r"turn (up|down) (?:the |your )?volume", text)
    if step:
        return "step", 15 if step[1] == "up" else -15
    if text in ("louder", "speak louder", "be louder", "quieter", "speak quieter", "be quieter"):
        return "step", 15 if "louder" in text else -15
    if text in ("maximum volume", "max volume", "volume maximum", "volume max", "turn it all the way up"):
        return "set", 100
    if text in ("volume", "volume level", "what is your volume", "what's your volume",
                "what is the volume", "what's the volume"):
        return "query", None
    match = re.fullmatch(r"(?:(?:set|change) )?(?:your |the )?volume(?: to| at)? (.+)", text)
    if not match:
        return None
    number = re.sub(r" (?:percent|per cent)$", "", match[1]).strip()
    words = number.split()
    value = None
    if number.isascii() and number.isdigit():
        value = int(number)
    elif number in _ONES:
        value = _ONES.index(number)
    elif number in _TENS:
        value = _TENS[number]
    elif number in ("hundred", "one hundred", "a hundred"):
        value = 100
    elif len(words) == 2 and words[0] in _TENS and words[1] in _ONES[1:10]:
        value = _TENS[words[0]] + _ONES.index(words[1])
    return ("set", value) if value is not None and 0 <= value <= 100 else ("invalid", None)
