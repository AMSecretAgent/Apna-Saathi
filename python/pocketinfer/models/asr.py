import requests
import base64
import time
from subprocess import check_output

from pocketinfer.models import ensure_healthy




class Asr:
    def __init__(self, timeout: float = 20.0):
        # See Nmt: without this a wedged bhashini service hangs the demo loop.
        self.timeout = timeout

    def infer(self, wav_bytes: bytes, language: str):

        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        payload = {
            "language": language,
            "audio_base64": audio_base64
        }

        response = requests.post("http://localhost:11400/asr", json=payload,
                                 timeout=self.timeout)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"ASR inference failed: {response.text}")

    @classmethod
    def verify(cls, args):
        # Shared with the other two clients: same service, one wait. See
        # pocketinfer.models.ensure_healthy.
        ok, msg = ensure_healthy()
        return (True, "ASR service is available.") if ok else (False, msg)

    @classmethod
    def update(cls, args):
        return True, "OK"
