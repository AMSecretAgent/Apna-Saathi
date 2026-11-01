import time
import requests
from subprocess import check_output

from pocketinfer.models import ensure_healthy




class Tts:
    def __init__(self, timeout: float = 30.0):
        # Longer than ASR/NMT: synthesis time scales with text length, and a
        # long answer can legitimately take a while. See Nmt for why this exists.
        self.timeout = timeout

    def infer(self, text: str, language: str):
        payload = {
            "text": text,
            "language": language
        }

        response = requests.post("http://localhost:11400/tts", json=payload,
                                 timeout=self.timeout)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"TTS inference failed: {response.text}")

    @classmethod
    def verify(cls, args):
        # Shared with the other two clients: same service, one wait. See
        # pocketinfer.models.ensure_healthy.
        ok, msg = ensure_healthy()
        return (True, "TTS service is available.") if ok else (False, msg)

    @classmethod
    def update(cls, args):
        return True, "OK"
