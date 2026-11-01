"""Shaping LLM output into something safe to translate, speak and display.

Deliberately stdlib-only and free of hardware imports, so this runs (and is
tested) on any laptop. The application module imports pyaudio and opencv
transitively and cannot be executed off-device.
"""
import re

# Markers that the answer correctly deflected a medical question instead of
# answering it. Used to tag the 'refused' path and to report a refusal rate.
SAFETY = re.compile(r"\b(doctor|hospital|clinic|104|health\s*worker)\b", re.I)

# Markdown the model emits despite being told not to. Line-leading bullets are
# matched first (both "- item" and "1. item" forms), then inline markup anywhere.
_MD = re.compile(r"^[ \t]*(?:[-*•]\s+|\d+[.)]\s+)|[*_#`>\[\]]", re.M)

# Sentence terminators, including the Devanagari danda.
_SENT = re.compile(r"(?<=[.!?।])\s+")


def speakable(text: str, max_sentences: int = 2, max_chars: int = 240) -> str:
    """Collapse LLM output into something bounded and natural when read aloud.

    Four jobs at once, which is why it is worth its own function:
      - strips markdown the model produces despite the system prompt
      - collapses interior newlines to a single line (the UI text boxes, and on
        the serial-attached board variant the line-framed protocol, both assume
        one line per field)
      - caps sentence count, which bounds *three* downstream stages -- NMT input,
        TTS synthesis time, and screen space -- not just this one
      - caps absolute length as a backstop for a model that ignores both
    """
    t = _MD.sub(" ", text)
    t = " ".join(t.split())
    if not t:
        return ""
    t = " ".join(_SENT.split(t)[:max_sentences]).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return t


def _self_check() -> None:
    # Markdown, bullets and interior newlines all collapse to one clean line.
    got = speakable("- **PM-KISAN** pays Rs 6000.\n\n1. In three instalments.\nExtra.")
    assert "\n" not in got and "*" not in got, got
    assert got.startswith("PM-KISAN pays Rs 6000."), got

    # Sentence cap applies.
    assert speakable("One. Two. Three.", max_sentences=2) == "One. Two."

    # Devanagari danda counts as a sentence end.
    assert speakable("एक। दो।", max_sentences=1) == "एक।"

    # Hard length cap, and it must not end mid-word.
    long = speakable("alpha bravo " * 60, max_chars=50)
    assert len(long) <= 51, (len(long), long)
    assert not long.rstrip(".").endswith(("alph", "brav")), long

    # Degenerate inputs do not raise.
    assert speakable("") == ""
    assert speakable("   \n\n  ") == ""
    assert speakable("***") == ""

    # Safety detection drives the 'refused' path.
    assert SAFETY.search("Please see a doctor.")
    assert SAFETY.search("Call 104 for advice.")
    assert not SAFETY.search("PM-KISAN pays 6000 rupees per year.")

    print("textshape self-check OK")


if __name__ == "__main__":
    _self_check()
