"""Speech recognition for the languages the Bhashini ASR does not ship.

Bhashini's IndicConformer service has exactly two checkpoints, Hindi and Tamil,
and returns HTTP 500 for anything else. This is whisper.cpp with
large-v3-turbo, which hears all nine of the languages the device can speak, and
English, which until now had no recogniser at all.

It is not a replacement for IndicConformer on Hindi. Measured on the same
recordings: Conformer 0.17-0.31 s, Whisper 0.87 s warm and nine seconds cold,
with comparable accuracy. So Hindi and Tamil stay on the fast model and only the
languages it cannot hear come here. The routing lives in the application, in
_hear(); this file only knows how to talk to the server.

The server is whisper.cpp's own, on 127.0.0.1:11500, as a systemd unit
(tools/whisper.service). It holds the model on the GPU beside the language
model. That co-residency was measured, not assumed -- see MEASURED.md -- and it
leaves a few hundred megabytes free, which is why nothing else may be added to
the board without measuring again.

Same shape as models/asr.py so the application can treat the two alike.
"""
import logging
import time
from subprocess import check_output

import requests

URL = "http://127.0.0.1:11500"

# Whisper's own language codes. Everything the device can speak, plus English.
LANGUAGES = {"en", "hi", "ta", "bn", "gu", "kn", "mr", "pa", "te"}


def _wait(seconds):
    deadline = time.time() + seconds
    while True:
        try:
            if requests.get(URL + "/", timeout=2.0).status_code < 500:
                return True
        except Exception:
            pass
        if time.time() >= deadline:
            return False
        time.sleep(0.5)


class Whisper:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def infer(self, wav_bytes: bytes, language: str):
        if language not in LANGUAGES:
            raise ValueError("whisper: unsupported language %r" % language)
        r = requests.post(
            URL + "/inference",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            # temperature 0 is greedy; the fallback ladder whisper uses at
            # higher temperatures is where hallucinated sentences come from.
            data={"language": language, "response_format": "json",
                  "temperature": "0"},
            timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError("whisper inference failed: %s" % r.text[:200])
        return {"text": (r.json().get("text") or "").strip()}

    # whisper names languages in full; everything else here uses ISO codes.
    NAMES = {"hindi": "hi", "english": "en", "tamil": "ta", "bengali": "bn",
             "gujarati": "gu", "kannada": "kn", "marathi": "mr",
             "punjabi": "pa", "telugu": "te"}

    def detect(self, wav_bytes: bytes):
        """Which language was spoken. Returns (code or None, {code: prob}).

        Detection only, no transcript: 0.65 s on this board against 1.0 s
        for a full pass. The code is None when whisper's best guess is a
        language the device does not have; the probabilities it returns are
        the top few only, so a caller comparing two languages must treat a
        missing one as zero.
        """
        r = requests.post(
            URL + "/inference",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language": "auto", "response_format": "verbose_json",
                  "detect_language": "true"},
            timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError("whisper detect failed: %s" % r.text[:200])
        j = r.json()
        code = self.NAMES.get((j.get("detected_language") or "").lower())
        probs = {k: float(v) for k, v in (j.get("language_probabilities") or {}).items()}
        return code, probs

    @classmethod
    def verify(cls, args):
        # The model takes ten to twenty seconds to load at boot. Wait for it
        # rather than restart it mid-load, for the same reason the bhashini
        # clients wait: a restart throws the partial load away.
        if _wait(60.0):
            return True, "whisper server is available."
        try:
            check_output("systemctl restart whisper.service", shell=True)
        except Exception as e:
            return False, "whisper is down and could not be restarted: %s" % e
        if _wait(60.0):
            return True, "whisper server is available."
        return False, "whisper server did not become healthy within 60s."

    @classmethod
    def update(cls, args):
        return True, "OK"
