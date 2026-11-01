#!/usr/bin/env python3
"""Probe the on-device bhashini service and report what actually works.

Run this as the FIRST command after getting shell on the board. The stock
application passes uppercase "EN" to /nmt but lowercase "hi" to /asr and /tts,
which cannot both be the convention, and nothing documents which endpoint wants
which. Guessing costs hours; this costs a minute.

    python3 tools/probe_bhashini.py
    python3 tools/probe_bhashini.py --wav sample.wav      # also exercise ASR

Prints a table of every language-code spelling against every endpoint, so the
values that go into the application settings are measured rather than assumed.
"""
import argparse
import base64
import json
import sys
import time

import requests

BASE = "http://localhost:11400"

# Every spelling anyone might reasonably have used.
LANG_FORMS = ["hi", "HI", "hin", "hin_Deva", "ta", "TA", "tam", "tam_Taml", "en", "EN", "eng"]


def _try(label, fn):
    t0 = time.time()
    try:
        result = fn()
        dt = time.time() - t0
        print(f"  OK    {label:<22} {dt:6.2f}s  {result}")
        return True
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT {label:<20}")
    except requests.exceptions.ConnectionError as e:
        print(f"  DOWN  {label:<22} {str(e)[:60]}")
    except Exception as e:
        print(f"  FAIL  {label:<22} {type(e).__name__}: {str(e)[:70]}")
    return False


def probe_health():
    print("\n=== /health ===")

    def go():
        r = requests.get(f"{BASE}/health", timeout=10)
        return f"{r.status_code} {r.text[:80]}"
    return _try("health", go)


def probe_nmt():
    print("\n=== /nmt (which language code spelling is accepted?) ===")
    for src in ["EN", "en"]:
        for tgt in ["HI", "hi", "hin_Deva"]:
            def go(src=src, tgt=tgt):
                r = requests.post(f"{BASE}/nmt", timeout=30, json={
                    "text": "How much money do farmers get?",
                    "src_lang": src, "tgt_lang": tgt})
                r.raise_for_status()
                return json.dumps(r.json(), ensure_ascii=False)[:90]
            _try(f"nmt {src}->{tgt}", go)


def probe_tts():
    print("\n=== /tts ===")
    for lang in ["hi", "HI", "ta", "en"]:
        def go(lang=lang):
            r = requests.post(f"{BASE}/tts", timeout=60, json={
                "text": "नमस्ते" if lang.lower() == "hi" else "hello",
                "language": lang})
            r.raise_for_status()
            audio = r.json().get("audio_base64", "")
            return f"{len(base64.b64decode(audio))} bytes of audio"
        _try(f"tts {lang}", go)


def probe_asr(wav_path):
    print("\n=== /asr ===")
    if not wav_path:
        print("  skipped (pass --wav to test; must be 16kHz mono 16-bit)")
        return
    wav = open(wav_path, "rb").read()
    b64 = base64.b64encode(wav).decode()
    for lang in ["hi", "HI", "hin_Deva", "ta"]:
        def go(lang=lang):
            r = requests.post(f"{BASE}/asr", timeout=60,
                              json={"language": lang, "audio_base64": b64})
            r.raise_for_status()
            return json.dumps(r.json(), ensure_ascii=False)[:90]
        _try(f"asr {lang}", go)


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="16kHz mono 16-bit wav to send to /asr")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    BASE = args.base
    print(f"probing {BASE}")

    if not probe_health():
        print("\nService is not answering. Try:")
        print("  sudo systemctl status bhashini_models.service")
        print("  sudo systemctl restart bhashini_models.service")
        print("  journalctl -u bhashini_models -n 50")
        sys.exit(1)

    probe_nmt()
    probe_tts()
    probe_asr(args.wav)

    print("\nPut the spellings marked OK into the application's default_settings,")
    print("and note any endpoint that only accepts one case -- that asymmetry is")
    print("worth a line in the architecture document.")


if __name__ == "__main__":
    main()
