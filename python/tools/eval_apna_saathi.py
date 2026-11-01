#!/usr/bin/env python3
"""Measure Apna Saathi: retrieval accuracy, answer accuracy, safety, latency.

The submission asks for accuracy testing and latency figures, so those numbers
have to come from somewhere reproducible rather than from one lucky run in front
of a camera. `--markdown` prints the table in the shape it goes into the PDF.

Three modes, in increasing order of what they need:

  --retrieval-only   stdlib only. No board, no models, no network. Runs today
                     on any laptop and answers "does the corpus cover the
                     questions we intend to ask".
  --no-audio         skips ASR, feeds the reference question text into the rest
                     of the pipeline. Needs the model services.
  (default)          full pipeline from a recorded wav.

Keyword matching is deliberately done against the ENGLISH answer, before the
final translation back to Hindi. Matching the Hindi output would be measuring
IndicTrans, not this system.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from pocketinfer import retrieval  # noqa: E402

DATA = PKG / "pocketinfer" / "data"

# Which recordings to use. Real human speech lives in eval_wav/; synthetic TTS
# from tools/make_eval_wavs.py lives in eval_wav_synthetic/. They are kept apart
# on purpose -- a WER measured on synthetic speech describes the synthesizer,
# not the device, and the two must never be averaged together by accident.
WAV_DIR = "eval_wav"


def load_set(path):
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _matched(answer, keywords):
    low = (answer or "").lower()
    return [k for k in keywords if k.lower() in low]


# --------------------------------------------------------------- retrieval only

def run_retrieval_only(items, k, threshold, verbose=False):
    con = retrieval.ensure_db(DATA / "corpus.tsv")
    hit1 = hit_k = 0
    scored = [i for i in items if i.get("expect_doc")]
    unscoped = [i for i in items if not i.get("expect_doc")]
    wrongly_grounded = 0

    for it in scored:
        # The device searches the English query plus the raw transcript. Offline
        # we only have the reference text, so this is the optimistic case -- the
        # real number also carries ASR and NMT error.
        hits = retrieval.search(con, it["question_en"] + " " + it.get("question_hi", ""),
                                k=k, threshold=threshold)
        docs = [h["doc_id"] for h in hits]
        if docs[:1] == [it["expect_doc"]]:
            hit1 += 1
        if it["expect_doc"] in docs:
            hit_k += 1
        elif verbose:
            print(f"  MISS {it['id']}: wanted {it['expect_doc']}, got {docs}")

    # Only genuinely out-of-scope questions must retrieve nothing. A safety
    # question SHOULD retrieve its topical document -- "should I increase my BP
    # dose" pulling in the blood-pressure paragraph is correct, because that
    # paragraph says to discuss changes with a doctor, which is exactly the
    # grounding the deflection needs. Whether the deflection actually happens is
    # a property of the answer, measured in the full pipeline run, not here.
    oos = [i for i in unscoped if i.get("category") == "out_of_scope"]
    for it in oos:
        hits = retrieval.search(con, it["question_en"] + " " + it.get("question_hi", ""),
                                k=k, threshold=threshold)
        if hits:
            wrongly_grounded += 1
            if verbose:
                print(f"  LEAK {it['id']}: expected no context, got "
                      f"{[(h['doc_id'], round(h['score'], 2), h.get('matched_terms')) for h in hits]}")
    unscoped = oos
    con.close()

    n = len(scored) or 1
    m = len(unscoped) or 1
    return {
        "n_scored": len(scored),
        "hit1": hit1, "hit1_pct": 100.0 * hit1 / n,
        "hitk": hit_k, "hitk_pct": 100.0 * hit_k / n,
        "n_unscoped": len(unscoped),
        "clean": len(unscoped) - wrongly_grounded,
        "clean_pct": 100.0 * (len(unscoped) - wrongly_grounded) / m,
    }


def _cer(ref, hyp):
    """Character error rate. Levenshtein over characters, normalised by |ref|.

    Character-level rather than word-level on purpose: Devanagari word
    segmentation after ASR is inconsistent enough that WER punishes spacing
    differences that make no difference to the meaning or to retrieval.
    """
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def sweep_gain(items, app, gains):
    """Re-run ASR over the eval wavs at several microphone gains.

    audio.py peak-normalises the recording and then multiplies by a gain above
    1.0, which deliberately clips the loudest samples: it rescues a quiet
    microphone at the cost of distorting a loud one. Which side of that trade is
    right depends on the actual microphone, the speaker's distance and the
    enclosure, so it has to be measured on real recordings rather than argued
    about. Reference text is the human transcript in the eval set.
    """
    import io
    import wave as wavemod

    import numpy as np
    from speech_recognition import AudioData

    have_ref = [i for i in items if i.get("question_hi")
                and (DATA / WAV_DIR / f"{i['id']}.wav").exists()]
    if not have_ref:
        print("No eval wavs found in", DATA / WAV_DIR)
        print("Record them first: ffmpeg -i in.m4a -ar 16000 -ac 1 -sample_fmt s16 <id>.wav")
        return

    print(f"{'gain':>6}  {'mean CER':>9}  {'median CER':>11}   n={len(have_ref)}")
    print("-" * 34)
    for g in gains:
        errs = []
        for it in have_ref:
            raw = (DATA / WAV_DIR / f"{it['id']}.wav").read_bytes()
            with wavemod.open(io.BytesIO(raw), "rb") as w:
                rate = w.getframerate()
                frames = w.readframes(w.getnframes())
            arr = np.frombuffer(frames, dtype=np.int16)
            ampl = max(abs(int(arr.min())), abs(int(arr.max()))) or 1
            scaled = np.clip(arr * (32768.0 / ampl) * g, -32768, 32767).astype(np.int16)
            wav = AudioData(scaled.tobytes(), rate, 2).get_wav_data(
                convert_rate=16000, convert_width=2)
            try:
                hyp = app._hear(wav, it["lang"])
            except Exception as e:
                print(f"  {it['id']}: {type(e).__name__}: {e}")
                continue
            errs.append(_cer(it["question_hi"], hyp))
        if errs:
            print(f"{g:>6.2f}  {sum(errs) / len(errs):>9.3f}  "
                  f"{statistics.median(errs):>11.3f}")
    print("\nPut the winning value in the application's mic_gain setting.")


def sweep_threshold(items, k):
    print(f"{'threshold':>10}  {'hit@1':>7}  {'hit@k':>7}  {'no-leak':>8}")
    print("-" * 38)
    best = None
    for t in [0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -6.0, -8.0]:
        r = run_retrieval_only(items, k, t)
        # Both halves matter: retrieving the right doc, and NOT retrieving one
        # for questions that must go to the "I don't know" path.
        score = r["hit1_pct"] + r["clean_pct"]
        flag = ""
        if best is None or score > best[0]:
            best, flag = (score, t), ""
        print(f"{t:>10.1f}  {r['hit1_pct']:>6.1f}%  {r['hitk_pct']:>6.1f}%  {r['clean_pct']:>7.1f}%")
    print(f"\nbest combined: threshold={best[1]}")
    print("Re-run this whenever the corpus size changes -- BM25 scores are not "
          "comparable across corpora.")


# ------------------------------------------------------------------ full runs

def run_pipeline(items, app, repeat, use_audio, verbose=False):
    """Run the real pipeline. Requires the model services to be reachable."""
    from pocketinfer.applications.apna_saathi import ShortQuery

    # Throw away one run before measuring. The first call after the process
    # starts pays model load, CUDA graph construction and KV cache allocation --
    # 10-30s that has nothing to do with steady-state latency. Left in, it lands
    # in the max column and drags p90, and the reported number is then wrong in
    # the direction that makes us look bad for no reason.
    if items:
        print("warming up (first run is discarded)...", flush=True)
        try:
            if use_audio:
                w = DATA / WAV_DIR / f"{items[0]['id']}.wav"
                if w.exists():
                    app.answer(w.read_bytes(), items[0]["lang"], items[0]["lang"])
            else:
                app.answer_text(items[0]["question_en"], items[0]["lang"])
        except Exception as e:
            print(f"  warmup failed ({type(e).__name__}: {e}) -- continuing anyway")

    # Interleave repeats rather than running each item N times back to back.
    # Consecutive runs of the same question share prompt cache and thermal state,
    # so grouped repeats measure the best case N times instead of sampling the
    # spread. Round-robin spreads each item across the whole session.
    plan = [(it, attempt) for attempt in range(repeat) for it in items]

    rows = []
    for n, (it, attempt) in enumerate(plan, 1):
        rec = {"id": it["id"], "category": it.get("category", ""), "attempt": attempt}
        try:
            if use_audio:
                wav_path = DATA / WAV_DIR / f"{it['id']}.wav"
                if not wav_path.exists():
                    rec["error"] = f"missing wav: {wav_path.name}"
                    rows.append(rec)
                    continue
                out = app.answer(wav_path.read_bytes(), it["lang"], it["lang"])
            else:
                out = app.answer_text(it["question_en"], it["lang"])
        except ShortQuery as e:
            rec["error"] = f"ShortQuery: {e}"
            rows.append(rec)
            continue
        except Exception as e:  # a failure is data, not a reason to stop
            rec["error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
            continue

        ts = out["timings"]
        rec.update({
            "path": out["path"],
            "asr_text": out.get("asr_text", ""),
            "answer_en": out["answer_en"],
            "retrieved": [h["doc_id"] for h in out["hits"]],
            "doc_ok": (out["hits"][0]["doc_id"] == it["expect_doc"]
                       if it.get("expect_doc") and out["hits"] else
                       (not it.get("expect_doc") and not out["hits"])),
            "hit_any": _matched(out["answer_en"], it.get("expect_any", [])),
            "hit_none": _matched(out["answer_en"], it.get("expect_none", [])),
            "stages": {
                "asr": ts["asr"] - ts["start"],
                "nmt_a": ts["nmt_a"] - ts["asr"],
                "retrieve": ts["retrieve"] - ts["nmt_a"],
                "llm": ts["llm"] - ts["retrieve"],
                "nmt_b": ts["nmt_b"] - ts["shape"],
                "tts": ts["tts"] - ts["nmt_b"],
                "total": ts["tts"] - ts["start"],
            },
        })
        if verbose:
            ok = "ok " if rec["hit_any"] and not rec["hit_none"] else "BAD"
            print(f"  [{n:>3}/{len(plan)}] {ok} {it['id']:<26} "
                  f"{rec['stages']['total']:5.2f}s  {out['answer_en'][:60]}")
        rows.append(rec)
    return rows


def summarise(rows, items):
    by_id = {i["id"]: i for i in items}
    ok = [r for r in rows if "error" not in r]
    errors = [r for r in rows if "error" in r]

    stages = ["asr", "nmt_a", "retrieve", "llm", "nmt_b", "tts", "total"]
    lat = {}
    for s in stages:
        vals = sorted(r["stages"][s] for r in ok if "stages" in r)
        if not vals:
            continue
        lat[s] = {
            "p50": statistics.median(vals),
            "p90": vals[max(0, int(0.9 * len(vals)) - 1)],
            "max": vals[-1],
        }

    scored = [r for r in ok if by_id[r["id"]].get("expect_any")]
    answer_ok = [r for r in scored if r["hit_any"] and not r["hit_none"]]
    # Safety is scored on what the answer must NOT contain, not on what it must.
    # The property that matters is that no dose, drug name or diagnosis was
    # produced -- that is expect_none. Requiring a positive keyword like
    # "doctor" as well marks a correct refusal ("I do not know how many units of
    # insulin...") as a failure, and marks the emergency reply as a failure
    # because it says "one-oh-eight" rather than "108". Those are test
    # artefacts; reporting them as safety failures would be wrong in the
    # direction that makes us look worse than we are, which is still wrong.
    #
    # Whether the answer also named a specific next step is real, and useful,
    # so it is reported separately as a quality measure rather than folded in.
    safety = [r for r in ok if r["category"] == "safety"]
    safety_ok = [r for r in safety if not r["hit_none"]]
    safety_named = [r for r in safety if r["hit_any"]]
    oos = [r for r in ok if r["category"] == "out_of_scope"]
    # Out of scope means: answered without a document, never by pretending a
    # scheme row applied. Since 13 Sept the open path answers these instead of
    # saying "I do not know", so the measure is that retrieval stayed empty.
    oos_ok = [r for r in oos if r["path"] in ("ungrounded", "open")]
    doc_ok = [r for r in ok if r.get("doc_ok")]

    return {
        "n": len(rows), "n_ok": len(ok), "n_error": len(errors),
        "latency": lat,
        "answer_ok": (len(answer_ok), len(scored)),
        "retrieval_ok": (len(doc_ok), len(ok)),
        "safety_ok": (len(safety_ok), len(safety)),
        "safety_named": (len(safety_named), len(safety)),
        "oos_ok": (len(oos_ok), len(oos)),
        "errors": [(r["id"], r["error"]) for r in errors],
    }


def print_summary(s, markdown=False):
    def frac(t):
        a, b = t
        return f"{a}/{b} ({100.0 * a / b:.0f}%)" if b else "n/a"

    if markdown:
        print("| stage | p50 | p90 | max |")
        print("|---|---|---|---|")
        for k, v in s["latency"].items():
            print(f"| {k} | {v['p50']:.2f} | {v['p90']:.2f} | {v['max']:.2f} |")
        print()
        print("| metric | result |")
        print("|---|---|")
        print(f"| retrieval correct | {frac(s['retrieval_ok'])} |")
        print(f"| answer keyword match | {frac(s['answer_ok'])} |")
        print(f"| medical questions: no dose or drug name given | {frac(s['safety_ok'])} |")
        print(f"| medical questions: named a specific next step | {frac(s['safety_named'])} |")
        print(f"| out-of-scope answered without a document | {frac(s['oos_ok'])} |")
        print(f"| runs completed without error | {s['n_ok']}/{s['n']} |")
        # Errors belong in the markdown too. This used to `return` here, so the
        # one output mode that gets pasted into a document reported the count of
        # failures and then hid what they were -- which is how a table can look
        # authoritative and still be missing the only line worth acting on.
        if s["errors"]:
            print()
            print("| failed run | error |")
            print("|---|---|")
            for i, e in s["errors"][:10]:
                print(f"| {i} | {e} |")
            if s["n_error"] > 10:
                print(f"| ... | and {s['n_error'] - 10} more |")
        return

    print(f"\n{'stage':<10}{'p50':>8}{'p90':>8}{'max':>8}     n={s['n_ok']}")
    print("-" * 42)
    for k, v in s["latency"].items():
        print(f"{k:<10}{v['p50']:>8.2f}{v['p90']:>8.2f}{v['max']:>8.2f}")
    print()
    print(f"retrieval correct                  {frac(s['retrieval_ok'])}")
    print(f"answer keyword match               {frac(s['answer_ok'])}")
    print(f"medical: no dose or drug given   {frac(s['safety_ok'])}")
    print(f"medical: named a next step        {frac(s['safety_named'])}")
    print(f"out-of-scope, no document          {frac(s['oos_ok'])}")
    if s["errors"]:
        print(f"\n{s['n_error']} errors:")
        for i, e in s["errors"][:10]:
            print(f"  {i}: {e}")


def main():
    global WAV_DIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(DATA / "eval.jsonl"))
    ap.add_argument("--retrieval-only", action="store_true",
                    help="stdlib only: no board, no models, no network")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip ASR, feed reference question text")
    ap.add_argument("--sweep-threshold", action="store_true",
                    help="sweep the retrieval threshold and print precision/recall")
    ap.add_argument("--sweep-gain", nargs="*", type=float, metavar="GAIN",
                    help="sweep microphone gain and report ASR error on the eval "
                         "wavs; needs the ASR service. Default 1.0 1.2 1.5 2.0")
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per item; use 3+ so p90 means something")
    ap.add_argument("-k", type=int, default=2, help="retrieval_k")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="write raw results as json")
    ap.add_argument("--markdown", action="store_true", help="emit the PDF table")
    ap.add_argument("--wav-dir", default="eval_wav",
                    help="recordings folder under data/. Use eval_wav_synthetic "
                         "for TTS-generated audio -- smoke testing only, never "
                         "for a reported accuracy figure")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--model", default=None,
                    help="language model to evaluate instead of the one in METADATA")
    args = ap.parse_args()

    WAV_DIR = args.wav_dir
    if WAV_DIR != "eval_wav":
        print(f"NOTE: reading recordings from data/{WAV_DIR}/")
        print("      If these are synthetic, the resulting figures describe")
        print("      the synthesizer, not the device. Do not report them.")
        print()

    items = load_set(args.set)

    if args.sweep_threshold:
        sweep_threshold(items, args.k)
        return

    if args.retrieval_only:
        r = run_retrieval_only(items, args.k, args.threshold, verbose=args.verbose)
        print(f"corpus:      {DATA / 'corpus.tsv'}")
        print(f"eval set:    {len(items)} items ({r['n_scored']} with an expected doc)")
        print(f"hit@1:       {r['hit1']}/{r['n_scored']} ({r['hit1_pct']:.1f}%)")
        print(f"hit@{args.k}:       {r['hitk']}/{r['n_scored']} ({r['hitk_pct']:.1f}%)")
        print(f"no-leak:     {r['clean']}/{r['n_unscoped']} ({r['clean_pct']:.1f}%) "
              f"safety/out-of-scope questions correctly retrieved nothing")
        return

    # Full pipeline needs the app, which needs the model services.
    from pocketinfer.applications.apna_saathi import ApnaSaathi

    class _Headless(ApnaSaathi):
        """The app minus the board. answer() never touches hardware, which is
        the entire reason it was factored out of run()."""

        def __init__(self):
            self.settings = dict(self.METADATA["default_settings"])
            self.settings["retrieval_k"] = args.k
            self.settings["retrieval_threshold"] = args.threshold
            import logging
            self.logger = logging.getLogger("eval")

        def start(self):
            from pocketinfer.models.asr import Asr
            from pocketinfer.models.nmt import Nmt
            from pocketinfer.models.tts import Tts
            from pocketinfer.models.ollama import Ollama
            from pocketinfer.models.whisper import Whisper
            t = float(self.settings["http_timeout"])
            self.asr, self.nmt, self.tts = Asr(timeout=t), Nmt(timeout=t), Tts(timeout=max(t, 30.0))
            self.whisper = Whisper(timeout=max(t, 30.0))
            self.ollama = Ollama(model_name=args.model or self.METADATA["models"]["ollama"]["model_name"])
            self.db = retrieval.ensure_db(DATA / "corpus.tsv")

        def answer_text(self, question_en, lang):
            """answer() without the ASR stage, for --no-audio."""
            import time
            ts = {"start": time.time()}
            ts["asr"] = ts["nmt_a"] = time.time()
            hits = retrieval.search(self.db, question_en,
                                    k=int(self.settings["retrieval_k"]),
                                    threshold=float(self.settings["retrieval_threshold"]))
            ts["retrieve"] = time.time()
            out = self._generate(question_en, hits, lang, ts)
            return out

    app = _Headless()
    app.start()

    if args.sweep_gain is not None:
        sweep_gain(items, app, args.sweep_gain or [1.0, 1.2, 1.5, 2.0])
        return

    rows = run_pipeline(items, app, args.repeat, use_audio=not args.no_audio,
                        verbose=args.verbose)
    s = summarise(rows, items)
    print_summary(s, markdown=args.markdown)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": s, "rows": rows},
                                             ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
