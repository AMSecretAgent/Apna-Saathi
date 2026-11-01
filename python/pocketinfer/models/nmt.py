import time
import requests
from subprocess import check_output

from pocketinfer.models import ensure_healthy




class Nmt:
    def __init__(self, timeout: float = 20.0):
        # The bhashini service is a single-process Flask app doing GPU work. With
        # no timeout a wedged service hangs the application loop forever, which
        # during a live demo is worse than an exception: the caller can catch an
        # exception and recover, but a hang just freezes the screen.
        self.timeout = timeout

    def infer(self, text: str, source_lang: str, target_lang: str):
        payload = {
            "text": text,
            "src_lang": source_lang,
            "tgt_lang": target_lang
        }

        response = requests.post("http://localhost:11400/nmt", json=payload,
                                 timeout=self.timeout)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"NMT inference failed: {response.text}")

    @classmethod
    def verify(cls, args):
        # Shared with the other two clients: same service, one wait. See
        # pocketinfer.models.ensure_healthy.
        ok, msg = ensure_healthy()
        return (True, "NMT service is available.") if ok else (False, msg)

    @classmethod
    def update(cls, args):
        return True, "OK"
