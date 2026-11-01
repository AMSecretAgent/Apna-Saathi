"""Print every question the device answers, as it answers it.

For language testing with a phone as the speaker: run this over SSH, hold the
button, play the clip, and read what the device heard, what that became in
English, which document it grounded on, and what it said back. Paste HEARD and
ANSWER into a translator to judge them in a language nobody here reads.

    ssh ubuntu@<ip> python3 /home/ubuntu/pocket-infer-sw/python/tools/watch_queries.py
"""
import json
import sys
import time
from pathlib import Path

LOG = Path("/tmp/apna_saathi_logs/log.jsonl")


def show(line):
    d = json.loads(line)
    st = d.get("settings", {})
    stages = d.get("stages", {})
    mode = d.get("mode") or st.get("mode")
    lin, lout = d.get("lang_in"), d.get("lang_out")
    print()
    if mode == "translate":
        print("TRANSLATE %s -> %s" % (lin, lout))
        print("HEARD  :", d.get("query") or "(nothing usable: script guard or silence)")
        print("SAID   :", d.get("answer_out"))
    else:
        docs = [r["doc_id"] for r in d.get("retrieval", [])][:1]
        print("%s  %s" % ("SCAN" if mode == "ocr" else "ASK", lout or st.get("output_language")))
        print("%s  :" % ("READ " if mode == "ocr" else "HEARD"),
              d.get("query") or "(nothing usable: script guard or silence)")
        if d.get("query_en") != d.get("query"):
            print("IN EN  :", d.get("query_en"))
        print("PATH   :", d.get("path"), ("from " + docs[0]) if docs else "")
        print("ANSWER :", d.get("answer_out"))
    # processing is release-to-answer, the number the device owns. The
    # spoken reply and the hold are not in it.
    print("TIME   : %.2f s release to answer  (asr %.2f, llm %.2f, tts %.2f)" % (
        stages.get("processing", 0), stages.get("asr", 0),
        stages.get("llm", 0), stages.get("tts", 0)))
    sys.stdout.flush()


def main():
    # The service writes this as root; we only read. It appears after the
    # first answered question of a boot.
    while not LOG.exists():
        print("no log yet at %s; ask one question" % LOG)
        time.sleep(2)
    print("watching %s; ask a question" % LOG)
    with LOG.open(encoding="utf-8") as f:
        f.seek(0, 2)  # only what happens from now on
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            try:
                show(line)
            except Exception as e:
                print("could not read a log line: %s" % e)


if __name__ == "__main__":
    main()
