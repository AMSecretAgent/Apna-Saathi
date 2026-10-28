# Third Party Notices

Apna Saathi is built on the Suno Sutra reference platform and bundles third
party models, fonts and libraries so that the device works with no network
connection. This file records what is included and under what terms.

The VYOMA challenge requires an open source prototype, and the 4 August office
hours left the redistribution question open. The short answer is that every
component below permits offline redistribution, and the obligations are
attribution ones. The longer answer is in the table.

## Our own code

Apna Saathi additions (`pocketinfer/applications/apna_saathi.py`,
`pocketinfer/retrieval.py`, `pocketinfer/textshape.py`, `tools/`, and the
`pocketinfer/data/` corpus) are released under the MIT License, matching the
upstream platform.

## Platform

| Component | Source | Licence | Obligation |
|---|---|---|---|
| Suno Sutra software (`pocketinfer`) | github.com/currentai-org/suno-sutra-sw | MIT | keep the copyright and permission notice |
| Suno Sutra hardware docs and CAD | github.com/currentai-org/suno-sutra-hw | MIT | same |

## Models

| Model | Role | Licence | Notes |
|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | answer generation | **Apache-2.0** | Deliberate choice. See "Why Qwen" below |
| **AI4Bharat IndicConformer** | ASR (Hindi, Tamil) | **MIT** | quantized build ships with the platform |
| **AI4Bharat IndicTrans2** | NMT (Indic ↔ English) | **MIT** | CTranslate2 int8 conversion; cite Gala et al. 2023, TMLR |
| **AI4Bharat Indic-TTS / flite voices** | TTS | **MIT** (AI4Bharat) / BSD-style (Flite) | per-voice provenance for the bundled `.flitevox` files should be confirmed against the Flite project |
| Vosk small English model | English ASR fallback | Apache-2.0 | unused in the shipped application path |
| Piper `en_US-lessac-medium` | English TTS fallback | MIT | unused in the shipped application path |

All model weights above are redistributable offline. Attribution is the only
obligation; none of them require a network callback, telemetry, or a licence
server, which is what makes a genuinely disconnected device possible.

### Why Qwen2.5-1.5B and not Gemma or Llama

This was decided on licence, not on benchmark scores.

- **Qwen2.5-1.5B is Apache-2.0** — an OSI-approved licence, with no use
  restrictions to pass downstream. It satisfies "open source" without argument.
- **Gemma 2** ships under the *Gemma Terms of Use*, which is not OSI-approved.
  It permits commercial use and redistribution, but attaches a Prohibited Use
  Policy that must be passed to every downstream recipient. For a prototype
  whose stated requirement is to be open source, and which is intended for
  deployment by government departments, that is a materially weaker position.
- **Llama 3.2** has the same issue under the Llama Community Licence, which
  additionally carries a monthly-active-user threshold.

Two cautions for anyone changing the model:

1. **Not all Qwen2.5 sizes are Apache-2.0.** The 3B and 72B variants are under
   the separate Qwen licence. "Upgrading" 1.5B → 3B silently changes the
   licensing position of the whole device.
2. Model size is capped by the challenge guidance at under 3B parameters, and
   by the memory budget at roughly one 2B-class model alongside the ~2 GB of
   resident Bhashini services.

## Fonts

| Font | Role | Licence |
|---|---|---|
| Noto Sans Devanagari | on-screen Hindi text | SIL Open Font License 1.1 |
| Fork Awesome | UI icons | SIL OFL 1.1 (fonts), MIT (code) |

The SIL OFL permits bundling and redistribution; the licence text must travel
with the font files, and the reserved font names must not be reused for
modified versions.

## Libraries

Runtime dependencies are listed in `python/requirements.txt`. The significant
ones are CTranslate2 (MIT), Ollama (MIT), OpenCV (Apache-2.0), NumPy (BSD-3),
PyAudio (MIT), SpeechRecognition (BSD-3), and the Adafruit Blinka / CircuitPython
display stack (MIT).

## Datasets

The corpus in `pocketinfer/data/corpus.tsv` was written by the team. It
paraphrases publicly published information about Indian government welfare
schemes and general public-health guidance. It contains no personal data and no
copied text from third party sources.

It is explicitly **not** medical advice: the application is built to refuse
dosage and diagnosis questions and to direct the user to a doctor or to the
national health helpline 104, and that behaviour is measured as a pass/fail
criterion in `tools/eval_apna_saathi.py`.

## Attribution text

For any redistribution of a built device or image, include this file together
with the MIT licence texts of the platform and the AI4Bharat models, the SIL OFL
text for the fonts, and the Apache-2.0 text for Qwen2.5.

Suggested citation for the translation models:

> Gala et al. (2023). *IndicTrans2: Towards High-Quality and Accessible Machine
> Translation Models for all 22 Scheduled Indian Languages.* Transactions on
> Machine Learning Research.
