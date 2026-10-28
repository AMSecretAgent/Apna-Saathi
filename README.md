# Apna Saathi

An offline household assistant for Indian families, built on the Suno Sutra
platform for the BHASHINI VYOMA Innovation Challenge.

Ask a question out loud in Hindi. Get a spoken answer in Hindi. No internet, no
account, no app, no data leaving the device.

Two domains: **government welfare schemes** (ration card, PM-KISAN, Ayushman
Bharat, pensions, Ujjwala) and **family health guidance** (medicine timing, ORS,
warning signs, where to get free checkups).

---

## Quick start

**Without any hardware** — works on any laptop, needs nothing but Python 3:

```bash
git clone <this-repo> && cd apna-saathi
./setup.sh
```

That fetches the platform source, applies our fixes to it, overlays our
application, and runs the self-checks. About a minute.

**With a Jetson reachable over SSH or Tailscale:**

```bash
export DEV=ubuntu@<name-or-ip>          # password: ubuntu
less docs/BOARD_BRINGUP.md              # ten ordered steps, ~40 min
./tools/deploy.sh $DEV
```

`docs/BOARD_BRINGUP.md` is the important one. Every step either answers
something the code currently has to guess at, or removes a known cause of a
mid-demo failure.

---

## What you can do without a device

Quite a lot, which is deliberate — all of this was built before the board was
reachable.

```bash
python3 pocketinfer/retrieval.py                        # self-check
python3 pocketinfer/textshape.py                        # self-check
python3 tools/eval_apna_saathi.py --retrieval-only -v   # does the corpus cover our questions?
python3 tools/eval_apna_saathi.py --sweep-threshold     # calibrate retrieval
```

The pipeline logic is separated from the hardware, so accuracy and corpus work
need no board, no models and no network.

---

## How it works

```
press and hold  ->  record  ->  release
  ASR       bhashini /asr     Hindi speech   -> Devanagari text
  NMT       bhashini /nmt     Hindi text     -> English query
  RETRIEVE  sqlite FTS5       English query  -> 0-2 grounding paragraphs
  LLM       ollama 1.5B       facts+question -> <=2 English sentences
  SHAPE     textshape.py      bounded, single line, no markdown
  NMT       bhashini /nmt     English        -> Hindi answer
  TTS       bhashini /tts     Hindi text     -> speech
```

**The model never sees Hindi.** We translate to English, reason in English, and
translate back, so the language model only has to be good at short English
instruction-following. That is what makes a 1.5B model viable, and it means
adding a language is a Bhashini model rather than a bigger LLM.

**Answers are grounded, not trusted.** A small model asked unaided what PM-KISAN
pays will produce a confident number that is sometimes wrong. Every answer comes
from a paragraph retrieved on device, and the screen names the source. With no
match the device says it does not know.

**Retrieval is SQLite FTS5 — no embedding model.** Under 5 ms, zero resident
memory. That memory belongs to the language model. Full reasoning, including the
two rules that stop a single shared word from grounding an answer, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Medical questions are refused, and that is tested.** Dosage and diagnosis
questions route to a doctor or the helpline 104, measured as a pass/fail number
rather than hoped for.

---

## Layout

```
setup.sh                          fetch platform, patch it, overlay our code
pocketinfer/
  applications/apna_saathi.py     the application
  retrieval.py                    FTS5 grounding      (self-check inside)
  textshape.py                    answer shaping      (self-check inside)
  data/corpus.tsv                 the knowledge base  <- accuracy lives here
  data/eval.jsonl                 36-item evaluation set
tools/
  eval_apna_saathi.py             accuracy + latency harness
  probe_bhashini.py               run this FIRST on a new board
  make_canned.py                  render the fallback phrases (on device)
  deploy.sh                       push to a device and restart
  bhashini_cloud.py               dev scaffolding only -- never shipped
patches/                          our fixes to the platform, + PR descriptions
docs/
  BOARD_BRINGUP.md                ordered runbook for a new device
  ARCHITECTURE.md                 technical design
  BOM.md  BUSINESS_PLAN.md  VIDEO_SCRIPT.md
vendor/suno-sutra-sw              platform source, fetched by setup.sh (not committed)
```

---

## Editing the corpus

This is where accuracy comes from, and it needs no Python.

`pocketinfer/data/corpus.tsv` is tab separated, one paragraph per row:

| column | purpose |
|---|---|
| `doc_id` | stable identifier, referenced by the eval set |
| `section` | `scheme` / `health` / `civic` — shown as the on-screen source label |
| `title` | short human-readable name |
| `aliases` | **where accuracy comes from**: English paraphrases, Hindi in Devanagari, romanised Hindi. Weighted highest |
| `body` | one self-contained paragraph, 40–90 words — it becomes prompt context, so length costs latency |

Two rules that matter more than they look:

1. **Carry the topic name in both scripts in every sibling document.** If one row
   has `kisan samman nidhi किसान सम्मान निधि` and its siblings have only the
   romanised form, the shared words stop cancelling out and the two-script row
   wins every question in the family — including ones about a facet it does not
   cover. Fixing this moved hit@1 from 74% to 100%.
2. **Put the words a real person would say into `aliases`**, not the words a
   civil servant would write.

The index rebuilds automatically when the TSV changes, so editing the file and
restarting the service is the whole deployment step.

After editing, re-run and read the misses — the harness names them:

```bash
python3 tools/eval_apna_saathi.py --retrieval-only -v
```

---

## Measuring

```bash
python3 tools/eval_apna_saathi.py --repeat 3 --markdown     # the table for the PDF
python3 tools/eval_apna_saathi.py --sweep-gain              # mic gain vs ASR error
python3 tools/eval_apna_saathi.py --sweep-threshold         # retrieval threshold
```

`--repeat 3` and the p90 column are not pedantry: a single-shot latency number on
a thermally varying board is not a measurement. The harness discards the first
run (model load) and interleaves repeats so it samples the spread rather than
measuring the best case three times.

**On reading the retrieval numbers.** `--retrieval-only` answers one narrow
question: does the corpus cover the questions we intend to ask? It runs on
reference text with no ASR or translation error, and the corpus was tuned against
that same set. It is a ceiling, not field accuracy. The honest numbers come from
the audio path and from users who did not write the corpus — report those
separately and say which is which.

---

## Bhashini cloud API

`tools/bhashini_cloud.py` exists only for development: reference transcripts,
and a cloud-versus-on-device comparison. **It must never be imported by anything
under `pocketinfer/`.** The device is required to work with no connectivity, the
demo is recorded with the radio off, and a network call in the runtime path
would defeat the entire premise.

Credentials come from the environment and are never committed:

```bash
export BHASHINI_USER_ID=...
export BHASHINI_API_KEY=...
```

---

## Licensing

MIT. Every bundled model is Apache-2.0 or MIT — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which also explains why the
language model is Qwen2.5-1.5B rather than a larger model under restrictive
terms, and which sizes you must not silently upgrade to.

Built on [suno-sutra-sw](https://github.com/currentai-org/suno-sutra-sw) by
Current AI (MIT). Our fixes to it are in `patches/`, offered upstream.
