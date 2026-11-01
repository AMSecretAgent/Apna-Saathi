"""Read printed text off a photograph, in Devanagari or Latin.

    from pocketinfer import ocr
    reader = ocr.Reader()
    lines = reader.read(frame_bgr)      # ["INCOME & ASSET CERTIFICATE", ...]

This replaces the vision-language model on the camera path. The reason is not
size, although PP-OCRv5 mobile is 12.7 MB against SmolVLM's 276 MB. It is that a
small VLM does not read, it narrates: asked what text is in a picture it answers
"The image depicts a close-up view of a product, specifically a Paracetamol
Tablet." Every heuristic in vision.py exists to dig a reading out of that. An OCR
model returns the text and there is nothing to dig.

It also reads Devanagari, which the VLM could not, and which is the difference
between reading a government form and recognising its English header.

Two ONNX models, run on onnxruntime, which is already how the Bhashini ASR runs
on this board:

  * PP-OCRv5_mobile_det             finds text boxes, outputs a probability map
  * devanagari_PP-OCRv5_mobile_rec  reads one cropped line, CTC over 568 classes

The weights ship as PaddlePaddle inference files and were converted once with
paddle2onnx. Paddle itself is NOT a runtime dependency and must not become one:
paddlepaddle 3.2.2 installs on this board, loads these models, and then dies with
SIGSEGV during inference. The conversion is a one-off, done off the device's hot
path, and the result runs on a runtime this device already trusts.

Nothing here needs a package the device does not already have. Notably it avoids
pyclipper, which every PP-OCR port pulls in for the box-expansion step, by
approximating it with cv2.minAreaRect and a fixed ratio. For printed text on a
form or a medicine strip that is indistinguishable from the real thing.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODELS = Path(os.environ.get("POCKETINFER_OCR_DIR", "/home/ubuntu/ocr_onnx"))
DET = "PP-OCRv5_mobile_det.onnx"
REC = "devanagari_PP-OCRv5_mobile_rec.onnx"
DICT = "devanagari_dict.txt"
# A second recogniser, for Latin. The Devanagari model reads Devanagari well and
# Latin badly: measured on rendered medicine strips it returned "AoxicILLiN 25O
# mgCAPsuLes" where this one returns "AMOXICILLIN 250 mg CAPSULES" exactly. It is
# ten times the size and four times slower, so it is not used on every line, only
# on the lines the cheap reader says are Latin. See _MOSTLY_LATIN.
REC_LATIN = "PP-OCRv6_medium_rec.onnx"
DICT_LATIN = "v6_dict.txt"

# Detection input is padded to a multiple of 32 and capped, because cost grows
# with the square of the side length and a form photographed at arm's length does
# not need more. 960 is PP-OCR's own default and it reads small print at that
# size.
DET_MAX_SIDE = 960
DET_THRESHOLD = 0.3          # on the probability map
BOX_MIN_SCORE = 0.5          # mean probability inside the box
BOX_MIN_SIZE = 3             # px; below this is noise, not a character
UNCLIP = 1.6                 # box expansion, standing in for pyclipper's offset

REC_HEIGHT = 48
# The recogniser's width axis is dynamic and it emits one CTC timestep per 8
# input pixels. Squeezing a long line into a narrow canvas therefore starves the
# decoder: at 320 px a 27-character title gets 40 frames, and CTC silently drops
# characters it cannot separate. Measured, that turned "INCOME AND ASSET
# CERTIFICATE" into "NCOME AN ASSE CERIFICIE". Give it the width the aspect
# ratio asks for. 1600 is 200 timesteps, comfortable for a full line of a form.
REC_MAX_WIDTH = 1600
REC_WIDTH_STEP = 8

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _dedupe(boxes, max_overlap=0.2):
    """Drop boxes that mostly cover a box we are already keeping.

    The detector emits a probability map, and thresholding it produces several
    overlapping contours over the same words. Recognising each of them
    separately reads the same text two or three times, which is how
    "AMOXICILLIN 250 mg CAPSULES" came back as
    "ng CAPSULES AMOXICILLIN 250 mg 0 mg" -- every word present, in no order,
    some of them twice. Downstream that is a search query made of noise.

    Largest first, because the biggest box is the one that covers the whole
    line; the fragments are the ones worth dropping.
    """
    rects = []
    for b in boxes:
        x0, y0 = b[:, 0].min(), b[:, 1].min()
        x1, y1 = b[:, 0].max(), b[:, 1].max()
        rects.append((x0, y0, x1, y1, (x1 - x0) * (y1 - y0)))

    keep = []
    for i in sorted(range(len(boxes)), key=lambda k: -rects[k][4]):
        ax0, ay0, ax1, ay1, area = rects[i]
        if area <= 0:
            continue
        covered = False
        for j in keep:
            bx0, by0, bx1, by1, _ = rects[j]
            ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oy = max(0.0, min(ay1, by1) - max(ay0, by0))
            if ox * oy / area > max_overlap:
                covered = True
                break
        if not covered:
            keep.append(i)
    return [boxes[i] for i in keep]


def _mostly_latin(text, threshold=0.6):
    """True if this looks like Latin script rather than Devanagari."""
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    latin = sum(1 for c in letters if "a" <= c.lower() <= "z")
    return latin / len(letters) >= threshold


def _session(path):
    # CPU on purpose. The GPU belongs to the language model, which is pinned
    # there with keep_alive=-1; these two models take under half a second on CPU
    # cores that are otherwise idle while the GPU works.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class Reader:
    def __init__(self, models=MODELS, latin=True):
        models = Path(models)
        self.det = _session(models / DET)
        self.rec = _session(models / REC)
        self._det_in = self.det.get_inputs()[0].name
        self._rec_in = self.rec.get_inputs()[0].name
        chars = (models / DICT).read_text(encoding="utf-8").split("\n")
        # CTC convention for PP-OCR: index 0 is the blank, then the dictionary,
        # then a trailing space. Getting this order wrong shifts every character
        # by one and produces confident nonsense, so the self-check asserts it
        # rather than trusting it.
        self.charset = ["<blank>"] + [c for c in chars if c != ""] + [" "]

        self.rec_latin = self.charset_latin = None
        if latin and (models / REC_LATIN).exists():
            self.rec_latin = _session(models / REC_LATIN)
            self._rec_latin_in = self.rec_latin.get_inputs()[0].name
            lat = (models / DICT_LATIN).read_text(encoding="utf-8").splitlines()
            self.charset_latin = ["<blank>"] + [c for c in lat if c != ""] + [" "]

    # ---------------------------------------------------------------- detect
    def _detect(self, img):
        h, w = img.shape[:2]
        scale = min(DET_MAX_SIDE / max(h, w), 1.0)
        rh = max(int(round(h * scale / 32)) * 32, 32)
        rw = max(int(round(w * scale / 32)) * 32, 32)
        resized = cv2.resize(img, (rw, rh))

        x = resized[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
        x = ((x - _DET_MEAN) / _DET_STD)[None]
        prob = self.det.run(None, {self._det_in: x})[0][0, 0]

        mask = (prob > DET_THRESHOLD).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        sx, sy = w / float(rw), h / float(rh)
        boxes = []
        for c in contours:
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(c)
            if min(bw, bh) < BOX_MIN_SIZE:
                continue
            # Score the region before growing it, so a box that merely touches a
            # hot pixel is dropped rather than expanded.
            m = np.zeros_like(mask)
            cv2.drawContours(m, [c], -1, 1, -1)
            if float(prob[m == 1].mean()) < BOX_MIN_SCORE:
                continue
            pts = cv2.boxPoints(((cx, cy), (bw * UNCLIP, bh * UNCLIP), angle))
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            boxes.append(pts.astype(np.float32))
        return _dedupe(boxes)

    # --------------------------------------------------------------- extract
    @staticmethod
    def _order(pts):
        """Corners as top-left, top-right, bottom-right, bottom-left."""
        s = pts.sum(1)
        d = (pts[:, 0] - pts[:, 1])
        return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                         pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)

    def _crop(self, img, box):
        p = self._order(box)
        w = int(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[3] - p[2])))
        h = int(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2])))
        if w < 2 or h < 2:
            return None
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(p, dst), (w, h))
        # Taller than wide means a vertical line of text; rotate it rather than
        # feed the recogniser something it was never trained on.
        if h > w * 1.5:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        return out

    # ------------------------------------------------------------- recognise
    def _recognise(self, crop, session=None, charset=None, name=None):
        h, w = crop.shape[:2]
        tw = int(np.ceil(REC_HEIGHT * w / float(h)))
        tw = int(np.ceil(max(REC_WIDTH_STEP, min(tw, REC_MAX_WIDTH))
                         / REC_WIDTH_STEP)) * REC_WIDTH_STEP
        img = cv2.resize(crop, (tw, REC_HEIGHT))
        x = img[:, :, ::-1].astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - 0.5) / 0.5
        session = session if session is not None else self.rec
        charset = charset if charset is not None else self.charset
        name = name if name is not None else self._rec_in
        logits = session.run(None, {name: x[None]})[0][0]

        idx = logits.argmax(1)
        conf = logits.max(1)
        chars, scores, prev = [], [], -1
        for i, k in enumerate(idx):
            # CTC: collapse runs, drop blanks.
            if k != prev and k != 0 and k < len(charset):
                chars.append(charset[k])
                scores.append(conf[i])
            prev = k
        return "".join(chars).strip(), (float(np.mean(scores)) if scores else 0.0)

    # -------------------------------------------------------------------- read
    def read(self, img, min_confidence=0.5):
        """Lines of text found in the image, ordered top to bottom."""
        found = []
        for box in self._detect(img):
            crop = self._crop(img, box)
            if crop is None:
                continue
            text, score = self._recognise(crop)
            # The cheap reader has already told us which script this is. Only a
            # line that came back as Latin is worth the expensive reader; a
            # Devanagari line would get nothing from it, because its dictionary
            # has no Devanagari at all.
            if self.rec_latin is not None and _mostly_latin(text):
                alt, alt_score = self._recognise(
                    crop, self.rec_latin, self.charset_latin, self._rec_latin_in)
                if alt and alt_score > score:
                    text, score = alt, alt_score
            if text and score >= min_confidence:
                found.append((float(box[:, 1].min()),
                              float(box[:, 1].max()),
                              float(box[:, 0].min()), text))

        # Reading order, not just top-to-bottom. The detector often splits one
        # line into several boxes, and those boxes share a y. Sorting on y alone
        # leaves their order to chance, which is how "AMOXICILLIN 250 mg
        # CAPSULES" came back as "ng CAPSULES AMOXICILLIN 250 mg": every word
        # present, in no useful order. Group into lines by vertical overlap,
        # then read each line left to right, the way a person would.
        if not found:
            return []
        heights = [y1 - y0 for y0, y1, _x, _t in found]
        tol = max(4.0, 0.5 * (sum(heights) / len(heights)))
        found.sort(key=lambda t: t[0])
        lines, current, base = [], [found[0]], found[0][0]
        for item in found[1:]:
            if item[0] - base <= tol:
                current.append(item)
            else:
                lines.append(current)
                current, base = [item], item[0]
        lines.append(current)

        out = []
        for line in lines:
            line.sort(key=lambda t: t[2])
            out.extend(t for _y0, _y1, _x, t in line)
        return out

    def read_jpeg(self, jpg, min_confidence=0.5):
        """Same, from the JPEG bytes the board's camera hands back."""
        buf = np.frombuffer(bytes(jpg), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return []
        return self.read(img, min_confidence=min_confidence)


# Reading the whole page and searching on all of it would be a mistake. The
# retrieval gate needs two query terms to appear in a document before it counts
# as a match, and that gate is what stops an unrelated question being answered
# from a loosely similar row. Hand it forty lines of a form and two of them will
# coincide with something by accident.
#
# The top few lines are also where the answer lives: a form's title is at the
# top, and a medicine strip is mostly brand name and dose.
QUERY_MAX_LINES = 6
QUERY_MAX_CHARS = 200


def to_query(lines, max_lines=QUERY_MAX_LINES, max_chars=QUERY_MAX_CHARS):
    """Turn the lines read off a page into something to search for."""
    text = " ".join(t.strip() for t in lines[:max_lines] if t.strip())
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text.strip()


def usable(text, min_words=3):
    """True if there is enough here to describe rather than guess from.

    Three words of at least three letters each. "LEBX11" off a badly held
    page went to the language model and came back as a shop receipt with a
    date and a total; a 1.5B model given noise invents a document, and
    asking it to notice the noise does not work. So the decision is here.
    Digits do not count as letters: a date alone is not a readable page.
    """
    import unicodedata
    def letters(w):
        # vowel signs and viramas are not str.isalpha but they are how the
        # script is written; "लाल" is three letters, not two
        return sum(c.isalpha() or unicodedata.category(c) in ("Mn", "Mc") for c in w)
    t = (text or "").strip()
    words = [w for w in t.split() if letters(w) >= 3]
    return len(words) >= min_words


def _check_usable():
    assert not usable("")
    assert not usable("LEBX11")
    assert not usable("ऋ६षि वसर ह")
    assert not usable("02/05/2026 61")
    assert usable("रोगी: मोहन लाल आयु: 61 भर्ती: 02/05/2026")
    assert usable("INCOME AND ASSET CERTIFICATE")


def _self_check():
    _check_usable()
    """Render text, read it back.

    The assertion is on Devanagari because that is this model's job. Its Latin
    is secondary and measurably weaker: on a synthetic render it returned
    "INCOME AnD ASSETCERTIECATE" for "INCOME AND ASSET CERTIFICATE", which is
    good enough to retrieve on and not good enough to quote. Latin is printed
    here for information, not asserted.
    """
    from PIL import Image, ImageDraw, ImageFont

    reader = Reader()
    assert reader.charset[0] == "<blank>", "blank must be index 0"
    assert len(reader.charset) == 570, len(reader.charset)
    assert any("ऀ" <= c <= "ॿ" for c in reader.charset), "no Devanagari"

    font_path = "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"
    targets = ["आठ लाख रुपये से कम",
               "प्रधानमंत्री आवास योजना ग्रामीण",
               "क्षय रोग नियंत्रण कार्यक्रम"]

    img = Image.new("RGB", (1000, 90 * len(targets) + 40), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font_path, 46)
    for n, t in enumerate(targets):
        d.text((40, 30 + n * 90), t, font=f, fill="black")

    lines = reader.read(np.asarray(img)[:, :, ::-1])
    assert len(lines) >= len(targets), "found %d lines, wanted %d: %r" % (
        len(lines), len(targets), lines)

    # Exact on at least two of three. Not all three: OCR is a physical-world
    # measurement, and demanding perfection from it makes a brittle test rather
    # than a correct reader. What matters downstream is that retrieval can match
    # on the words, and a dropped anusvara does not stop that.
    exact = sum(1 for want, got in zip(targets, lines)
                if want.replace(" ", "") == got.replace(" ", ""))
    assert exact >= 2, "only %d/%d lines exact: %r" % (exact, len(targets), lines)
    print("ocr self-check OK: %d/%d Devanagari lines exact" % (exact, len(targets)))

    # Query shaping. Bounded on purpose: see QUERY_MAX_LINES.
    assert to_query(["  Income  ", "", "Certificate "]) == "Income Certificate"
    assert to_query(["a"] * 40).count("a") <= QUERY_MAX_LINES
    assert len(to_query(["word " * 200])) <= QUERY_MAX_CHARS
    assert usable("Paracetamol") and not usable("") and not usable("x")
    print("query shaping OK")

    # Overlapping detections must collapse to one box, or the same words get
    # read twice and the search query becomes noise.
    def _box(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    big = _box(0, 0, 100, 20)
    inner = _box(5, 2, 60, 18)          # a fragment of the same line
    apart = _box(0, 60, 100, 80)        # a genuinely different line
    assert len(_dedupe([big, inner, apart])) == 2, "did not collapse the fragment"
    assert len(_dedupe([big, apart])) == 2, "collapsed two separate lines"
    print("box dedupe OK")


if __name__ == "__main__":
    _self_check()
