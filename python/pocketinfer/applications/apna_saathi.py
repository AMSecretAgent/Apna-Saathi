"""Apna Saathi -- an offline household assistant for government schemes and
family health guidance.

Pipeline, all on-device, no network:

    press  -> record -> release
      ASR      (bhashini /asr)      Hindi speech  -> Devanagari text
      NMT      (bhashini /nmt)      Hindi text    -> English query
      RETRIEVE (sqlite FTS5)        English query -> 0-2 grounding paragraphs
      LLM      (ollama, 2B)         facts + question -> <=2 English sentences
      SHAPE                         bounded, single line, no markdown
      NMT      (bhashini /nmt)      English       -> Hindi answer
      TTS      (bhashini /tts)      Hindi text    -> speech

The LLM sits between two translation hops and therefore never sees Indic text,
so it only has to be good at short English instruction-following. That is what
lets a 2B model do this job at all, and it is why the model choice below is the
smallest one that works rather than the largest one that fits.

Everything the pipeline does lives in answer(), which touches no hardware. run()
is only UI glue around it. That separation is what makes tools/eval_apna_saathi.py
possible without a board.
"""
import base64
import json
import logging
import threading
import re
import time
import wave
from io import BytesIO
from pathlib import Path
from subprocess import check_output

from pocketinfer import retrieval, safety, vision
from pocketinfer.textshape import speakable
from pocketinfer.applications.base import BaseApplication
from pocketinfer.applications.registry import RegisterApplication
from pocketinfer.audio import AudioPlayer
from pocketinfer.models.asr import Asr
from pocketinfer.models.whisper import Whisper
from pocketinfer.models.nmt import Nmt
from pocketinfer.models.tts import Tts
from pocketinfer.models.ollama import Ollama


DATA = Path(__file__).resolve().parent.parent / "data"
LOGDIR = Path("/tmp/apna_saathi_logs")

# Input and output are not the same set, because the hardware is not symmetric.
#
# ASR ships exactly two ONNX checkpoints, hi-conformer and ta-conformer, with
# their vocabularies hardcoded in the bhashini service. Adding a language there
# is an export plus a code change to a service we do not own.
#
# Output is bounded by flite's LANG_VOICE_MAP, which is much wider: nine voices
# are already installed on the card the organisers shipped. IndicTrans2 covers
# all of these and 25 more, so the translation hop was never the constraint.
#
# Three of these are verified by someone on the team; the rest are demonstrated
# capability. We do not claim a language nobody here can read.
# Two recognisers, routed by language, and the split is measured rather than
# chosen. Bhashini's IndicConformer ships Hindi and Tamil only and returns HTTP
# 500 for anything else. Whisper large-v3-turbo hears all nine, and English --
# which until it arrived had no recogniser at all, despite the settings page
# offering a button for it. On the same Hindi recordings Whisper's accuracy
# matched Conformer's but took 0.87 s against 0.17 s, so Hindi and Tamil stay
# on the fast model and only what it cannot hear goes to Whisper. See _hear().
CONFORMER_LANGS = {"hi", "ta"}

# Every layer of the language model on the GPU, always. Ollama otherwise
# "fits" the model to free memory at load time and keeps a 1 GiB margin; with
# bhashini and whisper resident that margin is not there, and it quietly moves
# a quarter of the layers to the CPU: 100% GPU became 29%/71% and the llm stage
# went from 1.4 s to 16 s. Measured on this board, 13 Sept. An explicit num_gpu
# skips the fit. Qwen2.5-1.5B has 29 layers; 99 means all of them.
LLM_GPU_LAYERS = 99

_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_NON_SPEECH = re.compile(r"\*[^*]*\*|\[[^\]]*\]|\([^)]*\)")

# Unicode block of each language's alphabet, for checking that a transcript
# is in the script it was asked for. Latin has no block here: English is
# accepted as it comes, and Latin loanwords inside an Indic sentence are
# tolerated by the ratio rather than by a rule.
_SCRIPT_BLOCK = {
    "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF), "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF),
}


def _in_script(text, lang):
    """True if text has at least as many characters in lang's alphabet as it
    has letters outside it. Vowel signs count for the alphabet (they are not
    str.isalpha, but they are how the script is written). English is Latin:
    Whisper once answered an English clip in Arabic script. An empty string
    passes; its emptiness is handled by the caller."""
    lo, hi = _SCRIPT_BLOCK.get(lang, (0x0041, 0x007A))
    own = sum(lo <= ord(c) <= hi for c in text)
    other = sum(c.isalpha() and not (lo <= ord(c) <= hi) for c in text)
    return own >= other
INPUT_LANGS = {"en", "hi", "ta", "bn", "gu", "kn", "mr", "pa", "te"}
OUTPUT_LANGS = {"bn", "gu", "hi", "kn", "mr", "pa", "ta", "te", "en"}
VERIFIED_LANGS = {"en", "hi", "ta"}

# flite calls Kannada "ka". Everything else here uses ISO, and "ka" is really
# ISO 639-1 for Georgian, so the flite spelling is confined to one lookup at the
# TTS boundary rather than allowed to leak into settings, filenames or logs.
_TTS_CODE = {"kn": "ka"}

# The panel has one font, NotoSansDevanagari, which covers Devanagari and Latin
# and nothing else. Tamil already renders as blank today for want of glyphs.
# For a language the font cannot draw, show the English answer and speak the
# local one: an honest fallback reads better than a row of empty boxes.
_SCREEN_LANGS = {"hi", "mr", "en"}

SYSTEM = (
    "You are Apna Saathi, a helpful assistant for Indian families. "
    "Answer using ONLY the FACTS provided. "
    "If the FACTS do not answer the question, say you do not know and suggest "
    "visiting the nearest Common Service Centre. "
    "Never diagnose illness and never give medicine doses; tell the user to see "
    "a doctor or call the free health helpline 104. "
    "Never tell the user that they personally are or are not eligible; you do "
    "not know their details, so explain who is eligible instead. "
    "Reply in at most 2 short sentences of simple English. "
    "Plain sentences only: no lists, no bullet points, no markdown, no line breaks."
)

GROUNDED = "FACTS:\n{ctx}\n\nQUESTION: {q}\nANSWER:"

# With no matching document the model answers on its own. This was a fixed
# "I do not know, ask at the Common Service Centre" until 13 Sept, when the
# owner heard it say that to "what is your name" and "should I go to college
# tomorrow" and asked for an assistant that answers. The medicine gate and the
# emergency reply stay in front of it; everything else is open.
SYSTEM_OPEN = (
    "You are Apna Saathi, a friendly assistant for Indian families. "
    "Answer the question directly and helpfully in at most 3 short sentences "
    "of simple English. Never give medicine doses; for anything serious say "
    "to see a doctor. "
    "Plain sentences only: no lists, no bullet points, no markdown, no line breaks."
)

OPEN = "QUESTION: {q}\nANSWER:"

# Text read off a paper. The shape the owner asked for after scanning a
# hospital discharge summary: the facts on the page, then one line of advice.
# OCR of a badly held page is noise, and a model given noise invents a
# document ("an ancient text, possibly the Rigveda" for three garbled
# syllables). Asking the model to notice noise did not work either: it mixed
# "hold it closer" into a real summary. ocr.usable() decides before the
# model sees anything, so this prompt has no conditions in it.
OPEN_DOC = (
    "Text read from a paper someone is holding up. The reading may have "
    "small mistakes.\n{q}\n\n"
    "In two or three short sentences say what this paper is, who it is for, "
    "and the main facts actually printed on it. Mention only facts you can "
    "see in the text above; never mention a date, amount or name that is not "
    "written there. Then end with "
    "one sentence of practical advice about what to do with it.\nANSWER:"
)

# A paper gets more room than a spoken answer: four sentences, so the facts
# and the advice both fit. TTS time scales with this and a scan is not the
# latency path the device is scored on.
PAPER_SENTENCES = 4
PAPER_LINES = 24
PAPER_TEXT_CHARS = 700
PAPER_CHARS = 420
PAPER_PREDICT = 120

# What each page asks. Hand-written for English and Hindi; the other seven
# go through the translator and are spoken, not shown, because the panel has
# no glyphs for them.
MENU_TEXT = {
    "welcome": {"en": "Welcome to Apna Saathi.",
                "hi": "नमस्ते, अपना साथी में आपका स्वागत है।"},
    "lang": {"en": "Which language do you prefer?",
             "hi": "आप कौन सी भाषा में सहज हैं?"},
    "task": {"en": "What would you like to do?",
             "hi": "आप क्या करना चाहते हैं?"},
    "voice": {"en": "Ask a question, or translate a conversation?",
              "hi": "सवाल पूछें, या बातचीत का अनुवाद करें?"},
    "xlate": {"en": "Translate into which language?",
              "hi": "किस भाषा में अनुवाद करें?"},
}

# Button order and names. Hindi and Marathi in their own script because the
# panel can draw Devanagari; the rest in English because it cannot draw theirs.
LANG_ORDER = ["hi", "en", "ta", "bn", "gu", "kn", "mr", "pa", "te"]
LANG_NAMES = {"hi": "हिंदी", "en": "English", "ta": "Tamil", "bn": "Bengali",
              "gu": "Gujarati", "kn": "Kannada", "mr": "मराठी", "pa": "Punjabi",
              "te": "Telugu"}

# The status bar draws over whatever was there before without clearing it, so a
# short string leaves the tail of a longer one visible underneath -- on screen
# that reads as "Ready - sourceReady - Press Buttonupport amount". Padding every
# write to a fixed width overwrites the remains. 34 characters is what fits on a
# 320px panel in the ASCII-only status font.
_STATUS_WIDTH = 34


# What the panel can actually show, measured on the device rather than guessed.
#
# Both TextBoxes are 320x100 at line_spacing 0.80, and the font bounding box is
# 29px tall, so the line pitch is 23px and exactly 4 lines fit. Anything past the
# fourth line is silently dropped by the blit. Wrapping itself is correct --
# adafruit_display_text 5.0.4 measures per-glyph shift_x, the same metric the
# renderer advances by -- so the only budget we control is how much text we hand
# it.
#
# Measured against the real corpus at 320px: Devanagari wraps at 41-48 chars per
# line, Latin at 32-40, because Devanagari matras are zero-advance. So the box
# holds roughly 160 characters of either script, and 140 leaves a line of margin.
#
# This was 90, which filled two of the four lines with Devanagari and read as a
# half-empty screen. The spoken answer was never affected and always complete;
# this only bounds the copy on screen, and ends it at a word with an ellipsis so
# a truncation looks deliberate rather than broken.
_SCREEN_CHARS = 140


def _for_screen(text, limit=_SCREEN_CHARS):
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "\u2026"



def _answer_for_screen(out, out_lang):
    """What to put on the panel: the local answer if the font can draw it.

    The spoken answer is always in the user's language. This only decides what
    the screen shows, and a row of blank boxes reads as a broken device while
    English reads as a deliberate fallback.
    """
    text = out["answer_out"] if out_lang in _SCREEN_LANGS else out["answer_en"]
    return _for_screen(text)


def _status(text):
    return text[:_STATUS_WIDTH].ljust(_STATUS_WIDTH)


def _silent_wav(seconds=0.5, rate=16000):
    """A short silent 16kHz mono wav, used only to force model load at startup."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(2 * int(rate * seconds)))
    return buf.getvalue()


class ShortQuery(Exception):
    """ASR returned nothing usable -- usually a button tap instead of a hold."""


@RegisterApplication({
    "name": "Apna Saathi",
    "description": "Offline household companion for government schemes and family "
                   "health, grounded in an on-device knowledge base.",
    "author": "Team Apna Saathi",
    "version": "0.1.0",
    "models": {
        # qwen3:1.7b since 13 Sept. Measured on this board beside bhashini and
        # whisper, same prompts: qwen2.5:1.5b added "liver cancer" to a
        # discharge summary that did not say it and told a certificate holder
        # to see a doctor; qwen3:1.7b reported what was printed. 100% GPU,
        # 34 tok/s against 37, about 110 MB more. qwen2.5:3b fitted with 275 MB
        # to spare, ran at 22 tok/s and suggested ibuprofen for a headache.
        # Qwen3 is Apache-2.0 as well; the history below still applies.
        #
        # qwen2.5:1.5b, and the licence is the deciding factor, not the benchmark.
        # This challenge requires an open source prototype. Qwen2.5 at 1.5B is
        # Apache-2.0; gemma2:2b ships under the Gemma Terms of Use, which is not
        # an OSI-approved licence and carries downstream use restrictions, and
        # llama3.2 has the same problem with the Llama Community Licence. Note
        # Qwen2.5 *3B* is NOT Apache-2.0 either -- only certain sizes are, so do
        # not "upgrade" to 3B without re-reading the licence.
        #
        # It is also not a VLM: the model only ever sees English (see the module
        # docstring), there is no camera here, and the organizers asked for under
        # 3B params and ideally 2B or fewer. A second model would not fit anyway
        # once the bhashini services have taken their ~2GB.
        # Ordered deliberately, and the order is a boot fix. The bhashini
        # clients wait up to 60s for their service to answer /health; ollama's
        # verify has no such wait. Verifying bhashini first hands ollama.service
        # that time to finish starting, so by the time we ask, it answers.
        "asr": {},
        "whisper": {},
        "nmt": {},
        "tts": {},
        # Options ride along so Ollama.verify() pins the model with the same
        # runner the first generate will ask for; otherwise boot loads it twice.
        "ollama": {"model_name": "qwen3:1.7b",
                   "options": {"num_ctx": 1024, "num_gpu": LLM_GPU_LAYERS}},
    },
    "default_settings": {
        "input_language": "hi",
        "output_language": "hi",
        "num_predict": 60,
        "num_ctx": 1024,
        "temperature": 0.3,
        "max_sentences": 2,
        "max_chars": 240,
        "retrieval_k": 2,
        # Calibrate with tools/eval_apna_saathi.py --sweep-threshold once the
        # corpus is at final size. BM25 scores scale with corpus size, so a value
        # measured on a small corpus will silently reject good hits on a big one.
        "retrieval_threshold": 0.0,
        "context_chars": 700,
        "mic_gain": 1.2,
        "http_timeout": 20.0,
        # Vision is off by default. It costs memory and seconds, and the
        # device is voice-first; this is an extra mode, not part of every
        # question. Turn on once install_vlm.sh has run.
        "vision_enabled": True,
        # 256M, not 500M, and the reason is placement rather than size. With the
        # text model pinned at num_ctx 1024, the 256M loads 100%% on GPU and reads
        # in about half a second; the 500M spills to 92%% CPU and takes seven,
        # leaving 403MB free instead of 1163MB.
        #
        # The 500M transcribes more completely -- "Paracetamol 500 mg Tablets IP"
        # against "PARACETAMOL" -- but the read feeds retrieval, so the drug NAME
        # is what matters and the dose is not something we would act on anyway:
        # dose questions are refused by design.
        "vision_model": "smolvlm-256m",
        # "ocr" or "vlm". OCR is PP-OCRv5 on onnxruntime: 12.7 MB against the
        # VLM's 276 MB, four lines in 0.44s against ~2.2s, and it reads
        # Devanagari, which the VLM cannot. The VLM path is kept because the
        # decision belongs to a photograph of a real medicine strip under real
        # light, not to a synthetic render, and that test happens on a bench
        # with the device in hand.
        "camera_reader": "ocr",
        # "assistant" answers questions from the knowledge base. "translate" is
        # a parent-teacher meeting: each hold of the button is translated and
        # spoken in the other language, direction alternating. No language
        # model, no retrieval, no safety gate -- it is a translator, not an
        # adviser, and it must not editorialise what a teacher said.
        #
        # Two-way only between languages that have ASR, which on this device is
        # Hindi and Tamil. Any of the nine output languages can be the listening
        # side; the speaking side is bounded by the two ASR checkpoints.
        "mode": "assistant",
        "translate_a": "hi",
        "translate_b": "ta",
        "vision_timeout": 60.0,
        # A quick tap reads the camera; holding records speech. There is
        # only one button, and a tap was previously a dead gesture -- it
        # produced "I did not catch that" and nothing else. Giving it a
        # meaning costs no hardware and turns a failure into a feature.
        "tap_seconds": 0.6,
    },
    "service_dependencies": ["ollama", "bashini_models"],
})
class ApnaSaathi(BaseApplication):

    def start(self):
        t = float(self.settings.get("http_timeout", 20.0))
        self.asr = Asr(timeout=t)
        self.whisper = Whisper(timeout=max(t, 30.0))
        self.nmt = Nmt(timeout=t)
        self.tts = Tts(timeout=max(t, 30.0))
        self.ollama = Ollama(model_name=self.METADATA["models"]["ollama"]["model_name"])

        # Constructed only when vision is enabled, so a device that never uses
        # the camera never touches the model at all.
        self.ocr = self.vlm = None
        if self.settings.get("vision_enabled"):
            if self.settings.get("camera_reader") == "ocr":
                try:
                    # Imported here, not at module scope: it pulls in cv2 and
                    # onnxruntime, and this file has to import on a laptop for
                    # tools/test_pipeline.py to mean anything.
                    from pocketinfer import ocr
                    self.ocr = ocr.Reader()
                except Exception:
                    # Missing model files must not stop the device booting. The
                    # voice path is the device; the camera is an extra.
                    self.logger.exception("OCR unavailable, camera reading off")
            else:
                self.vlm = Ollama(model_name=self.settings["vision_model"])

        self.db = retrieval.ensure_db(DATA / "corpus.tsv")
        LOGDIR.mkdir(parents=True, exist_ok=True)
        self._fail_streak = 0
        # One string, set by _warmup and read by the run loop. It was a
        # boolean with no readers, so the screen said "Ready" from the moment
        # the process started -- including through the ~53s ASR model load,
        # where a press does nothing visible for most of a minute.
        self._warm_state = "warming"
        # Which way the next translation goes: 0 is a -> b, 1 is b -> a.
        self._translate_dir = 0
        # Where the person is in the menus: lang, task, voice, xlate, or app.
        # The physical button only does something in "app".
        self._flow = "app"
        self._menu_items = []

        # Wipe whatever the previous run left behind. Without this, text from a
        # crashed instance stays on the panel and the new run draws over it --
        # on screen that looks like two messages fighting for the same row.
        try:
            self.board.clear_screen()
        except Exception:
            self.logger.exception("could not clear screen (non-fatal)")

        self.board.subscribe_to_ui(self.ui_cb)
        threading.Thread(target=self._warmup, daemon=True).start()
        super().start()

    @classmethod
    def verify_dependencies(cls):
        """Never let dependency verification stop the device from starting.

        service.py calls this before the app exists and does not catch anything,
        so a RuntimeError here is a process exit. Upstream raises one whenever a
        model fails to verify twice -- and with the wifi off, which is how this
        device is demonstrated, ollama's verify cannot tell "the daemon has not
        finished starting" from "the model is missing" and routes both to a pull
        that has no network to reach.

        A device that boots into a working button and reports a broken service
        on screen is strictly better than one that exits before anyone can see
        why. Warm-up sets the status text, so the failure stays visible.
        """
        try:
            return super().verify_dependencies()
        except Exception:
            logging.exception("dependency verification failed -- starting anyway")
            return True

    # ---------------------------------------------------------------- warm-up

    def _warmup(self):
        """Pay the model load cost at boot instead of on the first question.

        Ollama.verify() already pins the model with keep_alive=-1, but it sends
        no prompt, so the CUDA graph and KV cache are not built until the first
        real generate. The bhashini services likewise load their weights lazily.
        Roughly 15s here saves 5-10s on the first question -- which is the one
        being filmed.
        """
        failed = []
        self.board.statusbar(_status("Starting up - please wait"))
        try:
            self.ollama.generate(
                prompt="hi", system="Reply with OK.",
                options={"num_predict": 1, "num_ctx": self.settings["num_ctx"],
                         "num_gpu": LLM_GPU_LAYERS},
                keep_alive=-1, think=False)
        except Exception:
            self.logger.exception("LLM warmup failed (non-fatal)")
            failed.append("LLM")
        # ASR is the expensive one: the first call after the service starts pays
        # ~53s of model load, measured on this board. Every call after it is
        # under a second. Without this the first question of a demo hangs for a
        # minute, which is the single worst thing that could happen on camera.
        # A short burst of silence is enough to force the load; the transcript
        # is discarded.
        #
        # The load is longer than the HTTP timeout, and at a cold boot whisper
        # and the language model are loading at the same moment, so the first
        # attempt times out. The service keeps loading after the client hangs
        # up; a second call returns once it is done. Three attempts is a minute,
        # which covers the load with contention. Seen on a reboot, 13 Sept:
        # one attempt reported "limited: ASR" on a board that was fine.
        for attempt in range(3):
            try:
                self._hear(_silent_wav(), self.settings["input_language"])
                break
            except Exception:
                if attempt == 2:
                    self.logger.exception("ASR warmup failed (non-fatal)")
                    failed.append("ASR")
                else:
                    self.logger.info("ASR still loading, attempt %d", attempt + 1)
        # Whisper's first inference after boot is 2-3 s against 0.5 s warm.
        # The line above warmed Conformer only (the default input language is
        # Hindi). Not a failure if it is down: it carries no demo language.
        try:
            self.whisper.infer(_silent_wav(), "en")
        except Exception as e:
            self.logger.warning("whisper warmup failed (non-fatal): %s", e)

        for lang in {self.settings["input_language"], self.settings["output_language"]}:
            if lang == "en":
                continue
            try:
                # Translate then speak, which warms the exact NMT -> TTS chain a
                # real answer uses, and works for every output language instead
                # of the two that had greetings written into the source.
                hello = self.nmt.infer("Hello.", "EN", lang)["translated_text"]
                self._speak(hello, lang)
            except Exception:
                self.logger.exception("bhashini warmup failed for %s (non-fatal)", lang)
                failed.append(lang.upper())
        self._warm_state = "ready" if not failed else "limited: " + ",".join(failed)
        # Paint it here. The run loop writes the idle text once and then blocks
        # in wait_for_trigger_button_down(), so a state change during that block
        # is invisible until someone presses -- which is exactly backwards.
        self.board.statusbar(_status(self._idle_status()))
        self.logger.info("warmup finished: %s", self._warm_state)
        # Say hello and ask for a language. Hindi first, then English.
        for lang in ("hi", "en"):
            self._say_text(MENU_TEXT["welcome"][lang], lang)
        self._start_flow()

    # ------------------------------------------------------------- menus

    def _say_text(self, text, lang):
        """Speak a fixed line through TTS, and never let it break the flow."""
        try:
            self._play(self._speak(text, lang))
        except Exception as e:
            self.logger.warning("could not say %r in %s: %s", text, lang, e)

    def _menu_prompt(self, kind, lang):
        by_lang = MENU_TEXT[kind]
        return by_lang.get(lang) or self._localize(by_lang["en"], lang)

    def _show_menu(self, kind, items):
        """Put a question and its choices on the panel and say the question.

        items is a list of (value, label). The UI reports "MENU i" and the
        value at i is what it meant; the UI never learns the values.
        """
        self._flow = kind
        self._menu_items = list(items)
        lang = self.settings["output_language"]
        if kind == "lang":
            # Before a language is chosen the question is asked in two.
            title = MENU_TEXT["lang"]["en"] + "\n" + MENU_TEXT["lang"]["hi"]
        elif lang in _SCREEN_LANGS:
            title = self._menu_prompt(kind, lang)
        else:
            title = MENU_TEXT[kind]["en"]
        ui = getattr(self.board, "UI", None)
        if ui is not None:
            ui.show_menu(title, [label for _, label in self._menu_items])
        self.board.statusbar(_status("Choose on the screen"))
        if kind == "lang":
            for l in ("hi", "en"):
                self._say_text(MENU_TEXT["lang"][l], l)
        else:
            self._say_text(self._menu_prompt(kind, lang), lang)

    def _show_app(self):
        self._flow = "app"
        ui = getattr(self.board, "UI", None)
        if ui is not None:
            ui.show_app()
        self.board.top_text("")
        self.board.bottom_text("")
        self.board.statusbar(_status(self._idle_status()))

    def _start_flow(self):
        self._show_menu("lang", [(l, LANG_NAMES[l]) for l in LANG_ORDER])

    def _on_menu(self, value):
        """A choice was made on the page named by _flow."""
        if self._flow == "lang":
            if value not in INPUT_LANGS & OUTPUT_LANGS:
                return
            self.settings["input_language"] = value
            self.settings["output_language"] = value
            self.logger.info("language = %s", value)
            self._show_menu("task", [("ocr", "Scan paper"), ("voice", "Talk")])
        elif self._flow == "task":
            if value == "ocr":
                self.settings["mode"] = "ocr"
                self.logger.info("mode = ocr")
                self._show_app()
            elif value == "voice":
                self._show_menu("voice", [("ask", "Ask"), ("translate", "Translate")])
        elif self._flow == "voice":
            if value == "ask":
                self.settings["mode"] = "assistant"
                self.logger.info("mode = assistant")
                self._show_app()
            elif value == "translate":
                mine = self.settings["output_language"]
                self._show_menu("xlate", [(l, LANG_NAMES[l]) for l in LANG_ORDER if l != mine])
        elif self._flow == "xlate":
            if value not in OUTPUT_LANGS or value == self.settings["output_language"]:
                return
            self.settings["translate_a"] = self.settings["output_language"]
            self.settings["translate_b"] = value
            self.settings["mode"] = "translate"
            self._translate_dir = 0
            self.logger.info("mode = translate %s <-> %s",
                             self.settings["translate_a"], value)
            self._show_app()

    def _idle_status(self):
        """What the status bar says when the device is waiting for a press.

        Every warm-up stage is non-fatal by design, so a partial failure still
        reaches the run loop. Saying "Ready" there would be a lie, and the whole
        point of the warm-up is that the failure stays visible.
        """
        if self._warm_state == "ready":
            if self.settings.get("mode") == "translate":
                a, b = self.settings["translate_a"], self.settings["translate_b"]
                src, dst = (a, b) if self._translate_dir == 0 else (b, a)
                return "Translate " + src + " -> " + dst + " - hold"
            if self.settings.get("mode") == "ocr":
                return "Hold the paper up, press the button"
            return "Ready - Press Button"
        if self._warm_state == "warming":
            return "Starting up - please wait"
        return self._warm_state

    def _speak(self, text, lang):
        """Synthesise, and be the only place that knows flite's spelling."""
        return base64.b64decode(
            self.tts.infer(text, _TTS_CODE.get(lang, lang))["audio_base64"])

    def _localize(self, text_en, lang):
        """A fixed reply in the user's language.

        safety.py carries hand-written strings for a few languages only. Without
        this, selecting Bengali would speak the *English* string through a
        Bengali voice -- on the emergency path, which is the worst place in the
        device for it. Hand-written where we have it, translated where we do not.
        """
        if lang == "en" or lang in safety.EMERGENCY_REPLY:
            return text_en
        try:
            return self.nmt.infer(text_en, "EN", lang)["translated_text"].strip()
        except Exception:
            self.logger.exception("could not localise a fixed reply to %s", lang)
            return text_en

    def _pick_direction(self, wav_bytes, a, b):
        """Which of the two translate languages was just spoken, or None.

        Translate mode used to alternate strictly: a, then b, then a. One
        person testing alone spoke Hindi twice, the device expected English
        the second time, and whisper in forced-English mode translated the
        Hindi into English and passed that on as the transcript. Detection
        first; alternation only when detection cannot separate the two.
        """
        detect = getattr(self.whisper, "detect", None)
        if detect is None:
            return None
        try:
            code, probs = detect(wav_bytes)
        except Exception as e:
            self.logger.warning("language detection failed: %s", e)
            return None
        if code in (a, b):
            return code
        pa, pb = probs.get(a, 0.0), probs.get(b, 0.0)
        if pa == pb:
            return None
        return a if pa > pb else b

    def _hear(self, wav_bytes, lang):
        """Speech to text, on whichever recogniser can actually hear this
        language. Returns the transcript, stripped; empty if nothing usable."""
        if lang in CONFORMER_LANGS:
            return (self.asr.infer(wav_bytes, lang).get("text") or "").strip()
        text = (self.whisper.infer(wav_bytes, lang).get("text") or "").strip()
        # Whisper labels non-speech instead of staying silent: "*sad music*",
        # "[Music]", "(applause)". Seen 14 Sept, translated into Punjabi.
        text = _NON_SPEECH.sub("", text).strip()
        # Whisper's failure mode is not a bad transcript, it is a confident
        # sentence in the wrong alphabet: Latin gibberish for a Hindi clip,
        # "yst MIKE M.K. H.A." for Bengali. Seen on this board. Fed onward that
        # becomes a fluent wrong answer; dropped here it becomes "I did not
        # catch that", which is the honest reply.
        if not _in_script(text, lang):
            self.logger.info("whisper answered outside the %s script: %r", lang, text)
            return ""
        return text

    @staticmethod
    def _nmt_code(lang):
        """The Bhashini NMT service spells English in capitals and everything
        else in lowercase ISO. Indic to Indic goes through its third model."""
        return "EN" if lang == "en" else lang

    def translate(self, wav_bytes, src, dst, on_stage=None):
        """Speech in one language, out loud in another. Nothing else.

        Same return shape as answer() so the screen, the log and the eval
        harness treat it identically. `hits` is always empty and `path` is
        "translate", which is how a reader of the log tells the modes apart.
        """
        def stage(msg):
            if on_stage is not None:
                on_stage(msg)

        ts = {"start": time.time()}
        stage("1/3 Listening " + src)
        text = self._hear(wav_bytes, src)
        self.logger.info("TRANSLATE_ASR src=%s text=%r", src, text)
        ts["asr"] = time.time()
        if len(text) < 2:
            raise ShortQuery("nothing heard")

        stage("2/3 " + src + " -> " + dst)
        if src == dst:
            translated = text
        else:
            translated = self.nmt.infer(
                text, self._nmt_code(src), self._nmt_code(dst)
            )["translated_text"].strip()
        self.logger.info("TRANSLATE_NMT src=%s dst=%s output=%r", src, dst, translated)
        ts["nmt_a"] = ts["retrieve"] = ts["llm"] = ts["shape"] = ts["nmt_b"] = time.time()

        stage("3/3 Speaking " + dst)
        tts_bytes = self._speak(translated, dst)
        ts["tts"] = time.time()

        return {"asr_text": text, "query_en": text, "hits": [],
                "lang_in": src, "lang_out": dst,
                "source": src + " -> " + dst, "section": "translate",
                "answer_en": translated, "answer_out": translated,
                "tts_bytes": tts_bytes, "path": "translate", "timings": ts}

    # --------------------------------------------------------------- pipeline

    def answer(self, wav_bytes, in_lang, out_lang, on_stage=None):
        """Run the full pipeline. No board, no audio device, no screen.

        Returns a dict with the intermediate text at every stage, the retrieval
        hits, the synthesized audio, and per-stage timings. `path` is one of
        'grounded', 'ungrounded' or 'refused'.

        `on_stage` is an optional callback taking a short status string; run()
        passes board.statusbar, the eval harness passes nothing. Keeping it a
        callback is what stops this method from importing anything UI-shaped.
        """
        def stage(msg):
            if on_stage is not None:
                on_stage(msg)

        ts = {"start": time.time()}

        # --- 1. speech -> text
        stage("1/4 Understanding")
        asr_text = self._hear(wav_bytes, in_lang)
        ts["asr"] = time.time()
        if len(asr_text) < 3:
            raise ShortQuery(asr_text)

        # --- 1a. emergency short-circuit, before translation or any model
        # Checked on the raw transcript so it costs nothing and, more to the
        # point, so it still fires when the rest of the pipeline is broken.
        # Someone describing chest pain should not wait six seconds for a 2B
        # model to phrase a reply.
        if safety.is_emergency(asr_text):
            reply = self._localize(safety.emergency_reply(out_lang), out_lang)
            ts["nmt_a"] = ts["retrieve"] = ts["llm"] = ts["shape"] = time.time()
            stage("EMERGENCY")
            answer_out = reply
            tts_bytes = self._speak(answer_out, out_lang)
            ts["nmt_b"] = ts["tts"] = time.time()
            self.logger.warning("EMERGENCY path for query: %s", asr_text)
            return {
                "asr_text": asr_text, "query_en": asr_text, "hits": [],
                "source": "emergency", "section": "emergency",
                "answer_en": safety.emergency_reply("en"),
                "answer_out": answer_out, "tts_bytes": tts_bytes,
                "path": "emergency", "timings": ts,
            }

        # --- 2. -> English
        stage("2/4 Searching")
        if in_lang == "en":
            query_en = asr_text
        else:
            query_en = self.nmt.infer(asr_text, in_lang, "EN")["translated_text"].strip()
        ts["nmt_a"] = time.time()

        # Re-check after translation: the transcript may have been in a script
        # or phrasing the patterns miss, and the English form may not be.
        if safety.is_emergency(query_en):
            reply = self._localize(safety.emergency_reply(out_lang), out_lang)
            ts["retrieve"] = ts["llm"] = ts["shape"] = time.time()
            stage("EMERGENCY")
            tts_bytes = self._speak(reply, out_lang)
            ts["nmt_b"] = ts["tts"] = time.time()
            self.logger.warning("EMERGENCY path (post-translation): %s", query_en)
            return {
                "asr_text": asr_text, "query_en": query_en, "hits": [],
                "source": "emergency", "section": "emergency",
                "answer_en": safety.emergency_reply("en"),
                "answer_out": reply, "tts_bytes": tts_bytes,
                "path": "emergency", "timings": ts,
            }

        # --- 3. grounding
        # The raw Devanagari transcript is appended to the English query: FTS5
        # tokenizes Devanagari correctly and the corpus aliases column carries
        # Hindi keywords, so this still retrieves when the translation is poor.
        hits = retrieval.search(
            self.db, query_en + " " + asr_text,
            k=int(self.settings["retrieval_k"]),
            threshold=float(self.settings["retrieval_threshold"]))
        ts["retrieve"] = time.time()

        out = self._generate(query_en, hits, out_lang, ts, on_stage=on_stage)
        out["asr_text"] = asr_text
        return out

    def _generate(self, query_en, hits, out_lang, ts, on_stage=None, paper=False):
        """Second half of the pipeline: facts + question -> spoken answer.

        Split out so the eval harness can drive it directly with reference text
        and skip ASR, which otherwise makes every measurement of the LLM and the
        corpus depend on microphone quality.
        """
        def stage(msg):
            if on_stage is not None:
                on_stage(msg)

        # --- 3a. gated refusal: a medicine question with no matching document
        # is a refusal, not a prompt. Handing "which antibiotic should I take"
        # to an ungrounded 1.5B model is exactly how a device invents a
        # prescription. Only medicine: schemes and general health questions
        # go to the open answer below, by the owner's decision on 13 Sept.
        # Reading a printed medicine name back off a discharge summary is not
        # prescribing, so a paper skips the gate; the paper prompt forbids
        # adding anything to what is printed.
        if not hits and not paper:
            domain = safety.gated_domain(query_en)
            if domain == "medicine":
                answer_out = self._localize(safety.gated_reply(domain, out_lang), out_lang)
                ts["llm"] = ts["shape"] = ts["nmt_b"] = time.time()
                stage("4/4 Speaking")
                tts_bytes = self._speak(answer_out, out_lang)
                ts["tts"] = time.time()
                self.logger.info("gated refusal (%s): %s", domain, query_en)
                return {
                    "asr_text": "", "query_en": query_en, "hits": [],
                    "source": f"refused: {domain}", "section": domain,
                    "answer_en": safety.gated_reply(domain, "en"),
                    "answer_out": answer_out, "tts_bytes": tts_bytes,
                    "path": "refused", "timings": ts,
                }

        # --- 4. answer
        stage("3/4 Thinking")
        if hits:
            ctx = retrieval.context_block(hits, max_chars=int(self.settings["context_chars"]))
            prompt = GROUNDED.format(ctx=ctx, q=query_en)
            path = "grounded"
        else:
            prompt = (OPEN_DOC if paper else OPEN).format(q=query_en)
            path = "open"

        resp = self.ollama.generate(
            prompt=prompt,
            system=SYSTEM if hits else SYSTEM_OPEN,
            options={
                "num_predict": PAPER_PREDICT if paper else int(self.settings["num_predict"]),
                "num_ctx": int(self.settings["num_ctx"]),
                "num_gpu": LLM_GPU_LAYERS,
                # A paper is transcription, not conversation: temperature 0.
                # At 0.3 the same scan of a prescription came back once as
                # "fever and cough" and once as two invented syrup names.
                "temperature": 0.0 if paper else float(self.settings["temperature"]),
                "top_k": 20,
                "repeat_penalty": 1.05,
                # Cheap latency win, not cosmetics: decode stops the moment the
                # model starts a second paragraph or echoes the template back.
                "stop": ["\n\n", "QUESTION:", "FACTS:"],
            },
            keep_alive=-1,
            # No reasoning block: seconds of tokens nobody hears.
            think=False)
        ts["llm"] = time.time()

        # think=False should mean no reasoning block; strip one anyway, since
        # a spoken "<think>" is the worst way to find out it did not.
        # Qwen3 sometimes opens with a stray "/" (seen: "/No joke to answer.").
        answer_en = speakable(_THINK.sub("", resp.response).lstrip("/ "),
                              max_sentences=PAPER_SENTENCES if paper else int(self.settings["max_sentences"]),
                              max_chars=PAPER_CHARS if paper else int(self.settings["max_chars"]))

        # Last line of defence, and it exists because the model failed this on
        # real recordings twice: it told a user it was fine to stop an antibiotic
        # when the retrieved paragraph said the opposite, and it called an
        # invented drug name safe in pregnancy. Grounding did not prevent either.
        #
        # So if the question was medical and the answer vouches for safety, the
        # answer is discarded and replaced with the refusal. This will sometimes
        # throw away a correct answer. That is the acceptable direction to fail.
        domain = None if paper else safety.gated_domain(query_en)
        medical_doc = bool(hits) and hits[0]["section"] == "health"
        if (domain in ("medicine", "health") or medical_doc) and safety.asserts_safety(answer_en):
            self.logger.warning("discarded unsafe assertion (%s): %s", domain, answer_en)
            if medical_doc:
                # A vetted document matched: say what it says, in its own
                # words, instead of refusing. The model is the part that
                # inverted it.
                answer_en = speakable(hits[0]["body"], max_sentences=2,
                                      max_chars=int(self.settings["max_chars"]))
                path = "grounded"
            else:
                answer_en = safety.gated_reply(domain or "health", "en")
                path = "refused"
        # A medical "I do not know" gets the medical next step, not the
        # Common Service Centre, which issues documents and does not look at
        # symptoms. This was removed with the schemes gate by mistake on
        # 13 Sept and the eval caught it: "BP dose change" was told to visit
        # a CSC.
        elif domain in ("medicine", "health") and safety.says_unknown(answer_en):
            answer_en = safety.gated_reply(domain, "en")
            path = "refused"
        ts["shape"] = time.time()

        # --- 5. back to the user's language
        stage("4/4 Speaking")
        if out_lang == "en":
            answer_out = answer_en
        else:
            answer_out = self.nmt.infer(answer_en, "EN", out_lang)["translated_text"].strip()
        ts["nmt_b"] = time.time()

        tts_bytes = self._speak(answer_out, out_lang)
        ts["tts"] = time.time()

        return {
            "asr_text": "",
            "query_en": query_en,
            "hits": hits,
            "source": hits[0]["title"] if hits else "",
            "section": hits[0]["section"] if hits else "",
            "answer_en": answer_en,
            "answer_out": answer_out,
            "tts_bytes": tts_bytes,
            "path": path,
            "timings": ts,
        }

    def read_image(self, out_lang, on_stage=None):
        """Read printed text held up to the camera, then answer about it.

        The camera produces a *query*, not an answer. Whatever the vision model
        reads goes through the same retrieval and the same safety gates as a
        spoken question, for the same reason: a 500M model misreading a medicine
        name is the same failure as the speech model hearing "penicillin" for
        "insulin", and it has to fail closed in exactly the same way.
        """
        def stage(msg):
            if on_stage is not None:
                on_stage(msg)

        ts = {"start": time.time()}

        stage("1/4 Looking")
        jpg = self.board.camera_frame_jpg()
        if jpg is None:
            raise ShortQuery("no camera frame")

        if self.ocr is not None:
            from pocketinfer import ocr
            lines = self.ocr.read_jpeg(bytes(jpg))
            # The whole page, not a search query: a discharge summary lists the
            # medicines below line six, and the old 6-line, 200-character cap
            # (sized for retrieval, which a paper no longer uses) cut them off.
            text = ocr.to_query(lines, max_lines=PAPER_LINES, max_chars=PAPER_TEXT_CHARS)
            legible = ocr.usable(text)
        else:
            read = self.vlm.generate(
                prompt=vision.READ_PROMPT,
                images=[bytes(jpg)],
                options={"num_predict": 80, "temperature": 0.1},
                # Not pinned. The text model is the one that must never unload;
                # this one is used occasionally and can pay its own load cost,
                # which is what keeps both inside the memory budget.
                keep_alive="5m")
            text = vision.clean(read.response)
            legible = vision.usable(text)
        ts["asr"] = time.time()

        self.logger.info("camera read: %s", text)
        if not legible:
            raise ShortQuery("nothing legible")

        # The read is already English, so there is no inbound translation.
        ts["nmt_a"] = time.time()
        query_en = text

        if safety.is_emergency(query_en):
            reply = self._localize(safety.emergency_reply(out_lang), out_lang)
            ts["retrieve"] = ts["llm"] = ts["shape"] = time.time()
            tts_bytes = self._speak(reply, out_lang)
            ts["nmt_b"] = ts["tts"] = time.time()
            return {"asr_text": text, "query_en": query_en, "hits": [],
                    "source": "emergency", "section": "emergency",
                    "answer_en": safety.emergency_reply("en"),
                    "answer_out": reply, "tts_bytes": tts_bytes,
                    "path": "emergency", "timings": ts}

        # No retrieval for a paper. A discharge summary shared two words with
        # the food-poisoning row and a scheme row and got their answers; the
        # corpus answers questions, and a paper is not a question. The paper
        # prompt describes what is printed instead.
        ts["retrieve"] = time.time()

        out = self._generate(query_en, [], out_lang, ts, on_stage=on_stage, paper=True)
        out["asr_text"] = text
        out["lang_in"], out["lang_out"] = "ocr", out_lang
        return out

    # ------------------------------------------------------------------- loop

    def run(self):
        self.logger.debug("Starting with settings: %s", self.settings)
        while self.running:
            try:
                self.board.statusbar(_status(self._idle_status()))
                self.board.wait_for_trigger_button_down()

                # On a menu page the button does nothing; the choice is on
                # the screen. Wait out the press so it is not read twice.
                if self._flow != "app":
                    self.board.wait_for_trigger_button_up()
                    self.board.statusbar(_status("Choose on the screen"))
                    continue

                scanning = self.settings.get("mode") == "ocr"
                self.board.statusbar(_status("Reading..." if scanning else "Listening..."))
                self.board.top_text("")
                self.board.bottom_text("")
                t0 = time.time()
                # No microphone in scan mode: the press is a shutter.
                if not scanning:
                    self.board.audio.start()
                self.board.wait_for_trigger_button_up()
                held = time.time() - t0
                if not scanning:
                    self.board.audio.stop()
                self.board.led_animation(1)

                # Scan mode: any press reads the paper. Otherwise a short
                # tap is a mistake, not a question.
                if scanning:
                    if not (self.settings.get("vision_enabled")
                            and (self.ocr or self.vlm)):
                        raise ShortQuery("scan, but vision is off")
                    out = self.read_image(
                        self.settings["output_language"],
                        on_stage=lambda m: self.board.statusbar(_status(m)))
                    self.board.top_text(_for_screen(out["asr_text"]))
                    self.board.bottom_text(_answer_for_screen(
                        out, self.settings["output_language"]))
                    self.board.led_animation(0)
                    self.board.statusbar(_status(
                        "src: " + out["source"] if out["hits"] else "read - no match"))
                    self._play(out["tts_bytes"])
                    self._log(t0, out)
                    self._fail_streak = 0
                    continue

                if held < float(self.settings["tap_seconds"]):
                    raise ShortQuery("tap, not a question")
                if not self.board.audio.frames:
                    raise ShortQuery("no audio captured")

                # 16kHz because that is what the ASR was trained on; the recorder
                # negotiates its own rate against the hardware and usually picks
                # 48kHz. See audio.py.
                wav = self.board.audio.to_audio_data(
                    gain=float(self.settings["mic_gain"])
                ).get_wav_data(convert_rate=16000, convert_width=2)

                if self.settings.get("mode") == "translate":
                    a, b = self.settings["translate_a"], self.settings["translate_b"]
                    self.board.statusbar(_status("1/3 Listening"))
                    detected_src = self._pick_direction(wav, a, b)
                    src = detected_src
                    if src is None:
                        src = a if self._translate_dir == 0 else b
                    dst = b if src == a else a
                    self.logger.info(
                        "TRANSLATE_DIRECTION a=%s b=%s detected=%s chosen=%s dst=%s",
                        a, b, detected_src, src, dst)
                    out = self.translate(
                        wav, src, dst,
                        on_stage=lambda m: self.board.statusbar(_status(m)))
                    # The other person is expected next; detection overrides
                    # this on the next turn if the same person speaks again.
                    self._translate_dir = 1 if src == a else 0
                    self.board.top_text(_for_screen(out["asr_text"]))
                    self.board.bottom_text(_answer_for_screen(out, dst))
                    self.board.led_animation(0)
                    nxt = (a, b) if self._translate_dir == 0 else (b, a)
                    self.board.statusbar(_status(
                        "next: " + nxt[0] + " -> " + nxt[1]))
                    self._play(out["tts_bytes"])
                    self._log(t0, out)
                    self._fail_streak = 0
                    continue

                out = self.answer(wav,
                                  self.settings["input_language"],
                                  self.settings["output_language"],
                                  on_stage=lambda m: self.board.statusbar(_status(m)))
                out["lang_in"] = self.settings["input_language"]
                out["lang_out"] = self.settings["output_language"]

                self.board.top_text(_for_screen(out["asr_text"]))
                self.board.bottom_text(_answer_for_screen(
                    out, self.settings["output_language"]))
                self.board.led_animation(0)
                # Naming the source on screen is one string, and it visibly
                # demonstrates that the answer is grounded rather than invented.
                self.board.statusbar(_status(
                "src: " + out["source"] if out["hits"] else "no match"))

                self._play(out["tts_bytes"])
                self._log(t0, out)
                self._fail_streak = 0

            except ShortQuery as exc:
                self.logger.info("ShortQuery reason: %s", exc)
                self.logger.info("query too short, asking again")
                self.board.led_animation(0)
                if self.settings.get("mode") == "ocr":
                    # "I did not hear you" is the wrong apology for a camera.
                    self.board.statusbar(_status("Could not read - hold it closer"))
                    lang = self.settings["output_language"]
                    self._say_text(self._localize(
                        "I could not read that. Hold the paper closer and press again.", lang), lang)
                else:
                    self.board.statusbar(_status("Ready - please hold the button"))
                    self._say_canned("didnt_hear")

            except KeyboardInterrupt:
                self.logger.info("Exit")
                self.board.clear_screen()
                self.running = False

            except Exception as e:
                self.logger.exception("Error in main application loop: %s", e)
                self._fail_streak += 1
                self.board.led_animation(0)
                self.board.statusbar(_status("Error - press again"))
                self._say_canned("one_moment")
                if self._fail_streak >= 2:
                    self._recover()
                time.sleep(1)

    # ---------------------------------------------------------------- helpers

    def _play(self, wav_bytes):
        wave_obj = wave.open(BytesIO(wav_bytes), "rb")
        with AudioPlayer(wave_obj.getframerate(), self.board.alsa_playback_device) as player:
            player.play(wave_obj.readframes(wave_obj.getnframes()))

    def _say_canned(self, name):
        """Play a pre-rendered phrase.

        Pre-rendered rather than synthesized on demand because the failures this
        recovers from include "TTS is down" -- a recovery path that calls the
        broken component is not a recovery path.
        """
        p = DATA / "canned" / f"{name}_{self.settings['output_language']}.wav"
        if not p.exists():
            # Silence is the wrong failure here: these phrases exist to tell
            # someone the device did not catch what they said. English in the
            # wrong language is still an answer; saying nothing looks broken.
            self.logger.warning("no canned %s for %s, using English",
                                name, self.settings["output_language"])
            p = DATA / "canned" / f"{name}_en.wav"
        if not p.exists():
            self.logger.warning("missing canned audio: %s", p)
            return
        try:
            with wave.open(str(p), "rb") as w:
                with AudioPlayer(w.getframerate(), self.board.alsa_playback_device) as pl:
                    pl.play(w.readframes(w.getnframes()))
        except Exception:
            self.logger.exception("canned playback failed")

    def _recover(self):
        """Two consecutive failures means a service is wedged, not a bad question."""
        self.logger.warning("restarting bhashini_models after %d failures", self._fail_streak)
        try:
            check_output("systemctl restart bhashini_models.service", shell=True)
        except Exception:
            self.logger.exception("service restart failed")
        self._fail_streak = 0
        threading.Thread(target=self._warmup, daemon=True).start()

    def _log(self, t0, out):
        """Append one JSON line per interaction.

        Same shape as the stock hear_the_world log so anything already parsing
        that keeps working; the retrieval and answer fields are additions. This
        file is where the latency table in the submission comes from.
        """
        ts = out["timings"]
        rec = {
            "id": int(t0 * 1000),
            "mode": self.settings.get("mode"),
            # The languages this turn used. In translate mode they alternate
            # and are not the assistant's input/output setting.
            "lang_in": out.get("lang_in"),
            "lang_out": out.get("lang_out"),
            "query": out["asr_text"],
            "query_en": out["query_en"],
            "answer_en": out["answer_en"],
            "answer_out": out["answer_out"],
            "path": out["path"],
            "retrieval": [{"doc_id": h["doc_id"], "score": h["score"]} for h in out["hits"]],
            "settings": dict(self.settings),
            "timestamps": ts,
            "stages": {
                "record": ts["start"] - t0,
                "asr": ts["asr"] - ts["start"],
                "nmt_a": ts["nmt_a"] - ts["asr"],
                "retrieve": ts["retrieve"] - ts["nmt_a"],
                "llm": ts["llm"] - ts["retrieve"],
                "shape": ts["shape"] - ts["llm"],
                "nmt_b": ts["nmt_b"] - ts["shape"],
                "tts": ts["tts"] - ts["nmt_b"],
                # Two different numbers, and confusing them costs us marks.
                #
                # `processing` is button-release to answer-ready. That is the
                # latency the device is responsible for, and the one the
                # challenge means by "true latency".
                #
                # `total` additionally includes however long the user held the
                # button and spoke, which is not something we control. It is
                # kept because it is what someone with a stopwatch sees, but it
                # must never be quoted as system latency.
                "processing": ts["tts"] - ts["start"],
                "total": ts["tts"] - t0,
            },
        }
        self.logger.info(
            "processing %.2fs (asr %.2f llm %.2f tts %.2f) | spoke %.1fs, "
            "wall %.2fs | path=%s",
            rec["stages"]["processing"], rec["stages"]["asr"],
            rec["stages"]["llm"], rec["stages"]["tts"],
            rec["stages"]["record"], rec["stages"]["total"], out["path"])
        # Recreate the directory rather than assume start() did: /tmp can be
        # cleaned between boot and a question, and losing the interaction log is
        # not a reason to fail an interaction that already succeeded.
        try:
            LOGDIR.mkdir(parents=True, exist_ok=True)
            with (LOGDIR / "log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            self.logger.exception("could not write interaction log (non-fatal)")

    def ui_cb(self, msg):
        """Handle settings-page buttons.

        Message format is set by the UI: 'ASR Hi', 'TTS Ta', 'Reset', ... The
        language arrives at offset 4 because of the space. Values are whitelisted
        because this is a trust boundary -- on the serial-attached board variant
        these strings come off a wire that also carries error tokens.
        """
        if msg == "Home":
            self._start_flow()
        elif msg.startswith("MENU "):
            try:
                i = int(msg[5:])
            except ValueError:
                return
            if 0 <= i < len(self._menu_items):
                self._on_menu(self._menu_items[i][0])
        elif msg == "Reset":
            self.logger.info("Reset")
            check_output(["systemctl", "restart", "pocketinfer"])
        elif msg == "Reboot":
            self.logger.info("Reboot requested")
            check_output(["systemctl", "reboot"])
        elif msg == "Shutdown":
            self.logger.info("Shutdown requested")
            check_output(["systemctl", "poweroff"])
        elif msg.startswith("MODE "):
            mode = msg[5:].lower().strip()
            if mode not in ("assistant", "translate"):
                self.logger.warning("ignoring unknown mode %r", msg)
                return
            self.settings["mode"] = mode
            self._translate_dir = 0
            self.logger.info("mode = %s", mode)
            self.board.statusbar(_status(self._idle_status()))
        elif msg.startswith("TRA ") or msg.startswith("TRB "):
            # Translate-mode languages. Both must be speakable; the one that
            # will be spoken INTO must also be hearable, which is enforced at
            # use time by ASR simply failing, so warn here rather than refuse.
            lang = msg[4:].lower().strip()
            if lang not in OUTPUT_LANGS:
                self.logger.warning("ignoring unsupported translate language %r", msg)
                return
            key = "translate_a" if msg.startswith("TRA") else "translate_b"
            self.settings[key] = lang
            if lang not in INPUT_LANGS:
                self.logger.warning("%s=%s has no ASR: it can be listened to, "
                                    "not spoken into", key, lang)
            self._translate_dir = 0
            self.logger.info("%s = %s", key, lang)
        elif msg.startswith("LANG "):
            # One tap sets both directions. The settings page (patch 09) has a
            # single language grid now that every speakable language is also
            # hearable; the separate ASR/TTS messages below still work.
            lang = msg[5:].lower().strip()
            if lang not in INPUT_LANGS & OUTPUT_LANGS:
                self.logger.warning("ignoring unsupported language %r", msg)
                return
            self.settings["input_language"] = lang
            self.settings["output_language"] = lang
            self.logger.info("language = %s", lang)
        elif msg.startswith("ASR") or msg.startswith("TTS"):
            lang = msg[4:].lower().strip()
            asr = msg.startswith("ASR")
            key, allowed = (("input_language", INPUT_LANGS) if asr
                            else ("output_language", OUTPUT_LANGS))
            if lang not in allowed:
                self.logger.warning("ignoring unsupported %s language %r",
                                    "input" if asr else "output", msg)
                return
            self.settings[key] = lang
            self.logger.info("%s = %s", key, lang)


# Text shaping is covered by pocketinfer/textshape.py and retrieval by
# pocketinfer/retrieval.py -- both run standalone on a laptop. The rest of this
# module needs real hardware or real services, and is exercised by
# tools/eval_apna_saathi.py.
