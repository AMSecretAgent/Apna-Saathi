"""Reading what is held up to the camera.

The device is voice-first and stays voice-first. This adds one thing: point it
at printed text -- a medicine strip, a scheme pamphlet, a government form -- and
have it read and explain what is written there.

Two rules shape the design, and both come from failures measured on real speech.

**Vision output is a query, never an answer.** A 500M model reading a medicine
name will get it wrong sometimes, exactly as the speech model turned
"इंसुलिन" into "पेंसिलिन". So what the camera produces is fed into retrieval and
into the same safety gates as a spoken question. It never reaches the user
unmediated, and it never lets the language model free-generate about a drug.

**It is a separate mode, not a step in every question.** Vision costs seconds
and memory. Making it part of the normal loop would slow down the ninety percent
of questions that are purely spoken, for the ten percent that are not.

Stdlib plus the existing Ollama client only -- no new runtime, no torch, and
nothing that can disturb the bhashini service.
"""
import re

# What we ask the vision model. Deliberately narrow: this is not "describe the
# scene", it is "read this". A small model asked an open question invents; asked
# a closed one it mostly transcribes.
# Short on purpose. A 256M model given four sentences of instruction starts
# reciting them instead of reading the picture -- measured: asked to "write the
# medicine name, scheme name or form title first", it replied "medicine name
# scheme name form title". Small models echo long prompts. One question works
# better than a careful specification.
READ_PROMPT = "What text is printed in this image?"

NOTHING = "NOTHING LEGIBLE"

# Openings a small VLM uses when it is narrating rather than transcribing. It
# still often contains the answer, so these are stripped, not used to reject.
_PREAMBLE = re.compile(
    r"^(the (image|picture|photo|label)( is| shows| depicts| appears| contains)?"
    r"[^.]*?[.:]|it (is|shows|appears)[^.]*?[.:]|this (is|shows)[^.]*?[.:])\s*",
    re.I)

# Instruction-shaped fragments. A model reciting its brief produces these; a
# photograph of a medicine strip does not.
_INSTRUCTIONY = re.compile(
    r"(if there (is|are)|write that|write it out|do not (describe|guess|write)"
    r"|reply with|respond with|exactly:)", re.I)

# Outright refusals. Nothing to salvage from these.
_REFUSAL = re.compile(
    r"^(i (cannot|can.t|am unable to)|there (is|are) no|no text|unable to|sorry)",
    re.I)

_QUOTED = re.compile(r'["\u201c\u2018\']([^"\u201d\u2019\']{3,80})["\u201d\u2019\']')

# A run of capitalised words, or one long capitalised word. This is what a
# product name or a scheme title looks like inside a sentence of prose, and it
# is the part worth keeping.
_NAMEY = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][A-Za-z]{1,})*|[A-Z]{3,}(?:\s+[A-Z]{2,})*)")

_STOPNAMES = {"the", "this", "that", "image", "picture", "photo", "label",
              "product", "package", "it", "there"}


def _echoes_prompt(text):
    """True if the reply is mostly the instruction handed back."""
    words = {w.strip(".,:;").lower() for w in (text or "").split() if len(w) > 3}
    if not words:
        return False
    prompt_words = {w.strip("?.,:;").lower() for w in READ_PROMPT.split() if len(w) > 3}
    prompt_words |= {"medicine", "scheme", "title", "form", "name", "legible", "nothing"}
    if len(words & prompt_words) / len(words) > 0.6:
        return True
    return bool(_INSTRUCTIONY.search(text or ""))


def extract(text):
    """Pull the label text out of whatever the model said.

    Asked "what text is printed here", a 256M model answers "The image depicts a
    product package labeled Paracetamol Tablets IP" -- the reading is correct and
    buried in narration. Rejecting that as a non-answer throws away the one part
    that matters, so instead: prefer anything quoted, else the capitalised runs
    that a product name or scheme title forms inside prose, else the sentence
    with its preamble stripped.
    """
    t = " ".join((text or "").split())
    if not t or _REFUSAL.match(t) or _echoes_prompt(t):
        return ""

    quoted = _QUOTED.findall(t)
    if quoted:
        return max(quoted, key=len).strip(" .,")

    names = [n for n in _NAMEY.findall(t)
             if n.split()[0].lower() not in _STOPNAMES and len(n) > 3]
    if names:
        # Longest run: "Paracetamol Tablets IP" beats "Paracetamol".
        return max(names, key=len).strip(" .,")

    # No quoted text and no name-shaped run means the model narrated rather than
    # read. Measured: "The tablet is metallic, with a shiny, reflective
    # surface..." and "The handwriting is in black ink..." both reached
    # retrieval as queries and came back grounded on whatever loosely matched.
    #
    # Falling back to the stripped sentence was the leak. Printed labels are
    # capitalised or quoted; prose about a photograph is neither. Returning
    # nothing sends this to "I did not catch that", which is honest.
    return ""


def usable(text):
    """True if we got something retrieval can actually use."""
    t = (text or "").strip()
    if not t or NOTHING.lower() in t.lower():
        return False
    return len(t.split()) >= 1 and len(t) >= 3


def clean(text, max_chars=200):
    """Extract, then bound. The result becomes a search query and prompt context."""
    t = extract(text)
    if len(t) > max_chars:
        t = t[:max_chars].rsplit(" ", 1)[0]
    return t.strip(' "\'')


def _self_check():
    # The real readings from the device: correct text buried in narration.
    assert clean('The image depicts a close-up view of a product, specifically a '
                 'Paracetamol Tablet.') == "Paracetamol Tablet"
    assert clean('The package is labeled as "Paracetamol Tablets IP." The label is '
                 'prominently displayed.') == "Paracetamol Tablets IP"
    assert "Paracetamol" in clean('The image is a medical label for a medication '
                                  'called Paracetamol.')
    assert "PRADHAN MANTRI" in clean("PRADHAN MANTRI AWAS YOJANA application form")

    # Nothing to salvage.
    assert clean("I cannot read the text in this image") == ""
    assert clean("There is no visible text") == ""
    assert clean(NOTHING) == ""
    assert clean("") == ""

    # Prompt echo, measured on the device.
    assert clean("medicine name scheme name form title") == ""
    assert clean("Medicine name, scheme name, or form title.") == ""
    assert clean("If there is a medicine name, a scheme name, or a form title, "
                 "write that first.") == ""

    # Narration with no name and no quotes is not a reading.
    assert clean("The tablet is metallic, with a shiny, reflective surface") == ""
    assert clean("The paper has a textured surface with visible text") == ""

    assert usable("Paracetamol Tablets IP")
    assert not usable("")
    assert not usable(NOTHING)

    assert len(clean("Ambulance " * 100, max_chars=50)) <= 50

    print("vision self-check OK")


if __name__ == "__main__":
    _self_check()
