#!/usr/bin/env python3
"""Render the fallback phrases the application plays when something goes wrong.

Run this ON THE DEVICE, once, and commit the resulting wav files:

    python3 tools/make_canned.py

The phrases are pre-rendered rather than synthesized on demand because the
failures they recover from include "the TTS service is wedged". A recovery path
that calls the component you are recovering from is not a recovery path.

They are rendered by the device's own TTS, not by the cloud API, for two
reasons: the voice must match the voice used for real answers, and a recording
that quietly used a better cloud voice would misrepresent the device.
"""
import base64
import sys
import wave
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pocketinfer.models.tts import Tts  # noqa: E402
from pocketinfer.models.nmt import Nmt  # noqa: E402
from pocketinfer.applications.apna_saathi import OUTPUT_LANGS, _TTS_CODE  # noqa: E402

OUT = HERE.parent / "pocketinfer" / "data" / "canned"

# Short, calm, and never blaming the user -- these play when the device has
# already failed once, to someone who may not be sure they used it correctly.
PHRASES = {
    "didnt_hear": {
        "hi": "माफ़ कीजिए, मैं सुन नहीं पाया। बटन दबाकर रखिए और फिर बोलिए।",
        "ta": "மன்னிக்கவும், என்னால் கேட்க முடியவில்லை. பொத்தானை அழுத்திப் பிடித்துப் பேசுங்கள்.",
        "en": "Sorry, I did not catch that. Please hold the button down and speak.",
    },
    "one_moment": {
        "hi": "एक क्षण रुकिए, मैं फिर से कोशिश कर रहा हूँ।",
        "ta": "ஒரு நிமிடம், நான் மீண்டும் முயற்சிக்கிறேன்.",
        "en": "One moment, I am trying again.",
    },
    "no_match": {
        "hi": "इसके बारे में मुझे अभी जानकारी नहीं है। नज़दीकी जन सेवा केंद्र पर पूछिए।",
        "ta": "இதைப் பற்றி எனக்கு இப்போது தெரியவில்லை. அருகிலுள்ள சேவை மையத்தில் கேளுங்கள்.",
        "en": "I do not know about that yet. Please ask at the nearest Common Service Centre.",
    },
}


def main():
    # Default to every language the device can speak, not the three that had
    # phrases written into this file by hand. A missing fallback clip means the
    # device says nothing at the moment it most needs to say something.
    langs = sys.argv[1:] or sorted(OUTPUT_LANGS)
    OUT.mkdir(parents=True, exist_ok=True)
    tts = Tts()
    nmt = Nmt()

    failures = 0
    for name, by_lang in PHRASES.items():
        for lang in langs:
            text = by_lang.get(lang)
            if not text:
                # No hand-written phrase for this language. Translate the
                # English one rather than skip: these are short, fixed, and
                # reviewed by ear before committing, which is the point of the
                # instruction at the end of this script.
                try:
                    text = nmt.infer(by_lang["en"], "EN", lang)["translated_text"].strip()
                    print(f"  (translated {name} -> {lang}: {text})")
                except Exception as e:
                    print(f"FAIL translate {name} -> {lang}: {type(e).__name__}: {e}")
                    failures += 1
                    continue
            dest = OUT / f"{name}_{lang}.wav"
            try:
                audio = base64.b64decode(
                    tts.infer(text, _TTS_CODE.get(lang, lang))["audio_base64"])
            except Exception as e:
                print(f"FAIL {dest.name}: {type(e).__name__}: {e}")
                failures += 1
                continue
            dest.write_bytes(audio)
            with wave.open(BytesIO(audio), "rb") as w:
                secs = w.getnframes() / float(w.getframerate())
            print(f"  {dest.name:<22} {len(audio):>8} bytes  {secs:4.1f}s")

    print(f"\nwrote to {OUT}")
    if failures:
        print(f"{failures} failed -- check: systemctl status bhashini_models.service")
        sys.exit(1)
    print("Listen to each one before committing. A fallback phrase that is itself")
    print("unintelligible makes a bad moment worse.")


if __name__ == "__main__":
    main()
