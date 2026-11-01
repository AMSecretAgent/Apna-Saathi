"""On-device grounding for Apna Saathi.

A sub-2B LLM will invent government scheme amounts if asked unaided, so every
answer is grounded in a small curated corpus. Retrieval is SQLite FTS5 with
BM25: `sqlite3` is stdlib, FTS5 is compiled in, and it costs no RAM and no
extra model on a device where the LLM and the Bhashini services are already
competing for 8 GB of unified memory.

The query reaching us has usually been machine-translated into canonical
English by IndicTrans, which emits domain vocabulary ("pension", "instalment")
rather than colloquial paraphrase, so lexical overlap with a corpus we wrote
ourselves is high by construction. The raw Devanagari transcript is appended to
the query as insurance against a bad translation -- unicode61 tokenizes
Devanagari correctly, so a Hindi query alone can still hit the right row.

ponytail: BM25 over a few hundred curated paragraphs. If the corpus ever grows
past a few thousand rows, or users start paraphrasing far outside the corpus
vocabulary, switch to embeddings + a vector index and pay the ~400 MB.
"""
import os
import sqlite3
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

# Column weights for bm25(). Order must match the CREATE VIRTUAL TABLE below.
# doc_id is UNINDEXED so it gets 0.0; aliases is weighted highest because that
# column is where the hand-written paraphrases and Hindi keywords live, and it
# is the cheapest place to fix a retrieval miss.
_BM25_WEIGHTS = "0.0, 1.0, 4.0, 6.0, 1.0"

_SCHEMA = f"""
CREATE VIRTUAL TABLE docs USING fts5(
    doc_id UNINDEXED, section, title, aliases, body,
    tokenize='unicode61 remove_diacritics 2'
);
"""

# Small and deliberately English-only: Devanagari tokens are never stopwords
# here, and an over-eager list would strip the very terms that carry the query.
# Two-letter terms that carry a topic. The under-three-characters filter exists
# to drop noise like "ki" and "ka", but it also dropped "tb", so "where do I get
# TB medicine" searched on "medicine" alone and landed on a row about not
# sharing prescriptions. Whitelist the ones that name a disease, a category
# or a state rather than lowering the bar for everything.
_SHORT_OK = {"tb", "sc", "st", "up", "mp", "pm"}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "will", "would", "should", "shall",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "about",
    "and", "or", "but", "if", "as", "that", "this", "these", "those",
    "get", "gets", "got", "give", "gives", "tell", "please", "want",
    "much", "many", "some", "any", "all", "there", "have", "has", "had",
    # Hindi function words, in both scripts. Added when the corpus reached 65
    # rows and "how do I invest money in bitcoin" started retrieving a crop
    # insurance row on the strength of "पैसा" and "कैसे" alone. These carry no
    # topic; every question has them. Content words in Devanagari are still
    # never stopped, which is the property the comment above protects.
    "में", "से", "को", "का", "की", "के", "है", "हैं", "और", "या", "कब", "कैसे",
    "कहाँ", "कहां", "क्या", "क्यों", "चाहिए", "मुझे", "मेरा", "मेरी", "मेरे",
    "kab", "kaise", "kahan", "kya", "kyun", "hai", "hain", "ka", "ki", "ke",
    "ko", "se", "mein", "me", "aur", "ya", "chahiye", "mujhe", "mera", "meri",
}

def _tokens(text: str) -> list[str]:
    """Split into words, keeping Indic combining marks attached.

    Not a regex: Python's `\\w` matches letters and digits but NOT Unicode
    combining marks (category M), so `re.findall(r"\\w+", "किसान")` returns
    ['क', 'स', 'न'] -- the matras are treated as separators and the word
    shatters into bare consonants. Same problem in Tamil. This costs a few
    lines and makes Hindi queries work at all.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isalnum() or unicodedata.category(ch)[0] == "M":
            buf.append(ch)
        elif buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def _fts_query(text: str) -> Optional[str]:
    """Turn free text into a safe FTS5 MATCH expression, or None.

    Every term is double-quoted so FTS5 operators that fall out of ASR or NMT
    output -- `*`, `OR`, `NEAR(`, an unbalanced quote -- are treated as literal
    text instead of raising OperationalError in the middle of a live demo.
    Terms are OR-joined: AND would reject any query containing one word absent
    from the corpus, which is most real questions.
    """
    terms = []
    for tok in _tokens(text.lower()):
        if (len(tok) < 3 and tok not in _SHORT_OK) or tok in _STOPWORDS:
            continue
        if tok not in terms:  # dedupe, preserve order
            terms.append(tok)
    if not terms:
        return None
    return " OR ".join('"%s"' % t for t in terms)


def default_db_path(tsv_path: Path) -> Path:
    """Where the built index lives.

    Deliberately not next to the TSV: the index is a build artifact that should
    not be committed, and on a development machine the repo may sit on a network
    or WSL share where SQLite cannot take file locks at all. The device has this
    on local disk and does not care.
    """
    # Namespaced by user id. The service runs as root under sudo while
    # development and test runs happen as the ordinary user, and a shared path
    # in /tmp means whoever builds the index first owns it -- the other then
    # fails with a bare PermissionError from a rename, a long way from the cause.
    try:
        uid = os.getuid()
    except AttributeError:      # not POSIX
        uid = 0
    return Path(tempfile.gettempdir()) / f"pocketinfer_{Path(tsv_path).stem}_{uid}.db"


def build(tsv_path: Path, db_path: Path) -> int:
    """(Re)build the FTS index from the TSV. Returns the row count."""
    tsv_path, db_path = Path(tsv_path), Path(db_path)
    rows = []
    with tsv_path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        expected = ["doc_id", "section", "title", "aliases", "body"]
        if header != expected:
            raise ValueError(f"{tsv_path} header is {header}, expected {expected}")
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f"{tsv_path}:{lineno} has {len(parts)} columns, expected 5")
            rows.append(parts)

    tmp = db_path.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        con.executescript(_SCHEMA)
        con.executemany("INSERT INTO docs VALUES (?,?,?,?,?)", rows)
        con.commit()
    finally:
        con.close()
    tmp.replace(db_path)  # atomic: a half-built index never becomes the live one
    return len(rows)


def ensure_db(tsv_path: Path, db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Rebuild if the index is missing or older than the TSV, then open it.

    Called from the application's start(), so editing corpus.tsv over SSH and
    restarting the service is the whole deploy story -- no separate ingest step
    to forget at 11 PM on submission day.
    """
    tsv_path = Path(tsv_path)
    db_path = Path(db_path) if db_path is not None else default_db_path(tsv_path)
    if not db_path.exists() or db_path.stat().st_mtime < tsv_path.stat().st_mtime:
        build(tsv_path, db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _terms(text: str) -> list[str]:
    """Content terms of a query, in order, deduped."""
    out = []
    for tok in _tokens(text.lower()):
        if (len(tok) >= 3 or tok in _SHORT_OK) and tok not in _STOPWORDS and tok not in out:
            out.append(tok)
    return out


def _distinct_words(matched, aliases):
    """How many different *words* a list of matched terms represents.

    The corpus convention writes every proper noun in both scripts, side by
    side: "delhi दिल्ली", "ews ईडब्ल्यूएस". The query does the same, because it
    is the English translation plus the raw transcript. So a question that
    merely mentions Delhi hands the Delhi row two matching terms -- "delhi" and
    "दिल्ली" -- and the two-term gate waves it through. That is how "when is
    the train from Delhi to Patna" came to retrieve an EWS application row.

    A Latin term and a Devanagari term that sit next to each other in the
    aliases are one word, not two. Collapse them before counting.
    """
    # Direction matters. The convention is Latin first, then its Devanagari:
    # "bihar बिहार". Checking both directions paired "बिहार" with "apply" merely
    # because "bihar बिहार apply" puts them side by side, and collapsed a row's
    # own distinguishing terms. Each Devanagari term pairs at most once.
    # A pair is one word, so it counts once: every Latin term counts, and a
    # Devanagari term counts only if no Latin term absorbed it.
    latin = [t for t in matched if t.isascii()]
    unpaired = [t for t in matched if not t.isascii()]
    for t in latin:
        hit = next((d for d in unpaired if f"{t} {d}" in aliases), None)
        if hit is not None:
            unpaired.remove(hit)
    return len(latin) + len(unpaired)


def search(con: sqlite3.Connection, text: str, k: int = 2,
           threshold: float = 0.0, min_terms: int = 2) -> list[dict]:
    """Top-k rows scoring at or below `threshold` (bm25 is negative, lower is better).

    The default 0.0 means "accept anything FTS5 matched at all" -- off-topic
    queries already return nothing because no term matches, so the threshold is
    only there to reject *weak* matches.

    It is a calibrated knob, not a constant: BM25 scores scale with corpus size
    and IDF, so a good hit scores about -1.0 on a 3-row corpus and about -6 on a
    realistic one. Sweep it with tools/eval_apna_saathi.py --sweep-threshold
    once the corpus is at its final size and put the result in the app's
    default_settings. Never hardcode a number measured on a different corpus.
    """
    match = _fts_query(text)
    if match is None:
        return []
    try:
        cur = con.execute(
            f"SELECT doc_id, section, title, aliases, body, "
            f"bm25(docs, {_BM25_WEIGHTS}) AS score "
            "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
            # Over-fetch generously. Rows are re-ranked below by term coverage,
            # so anything BM25 buries is invisible to that re-ranking -- and BM25
            # buries exactly the cases we care about, where several documents
            # share a scheme name and differ only in facet. Retrieval is a
            # sub-millisecond scan over a few hundred rows, so the width is free.
            (match, max(k * 10, 30)),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        # Belt and braces: _fts_query already quotes every term, but a retrieval
        # failure must never take down the demo loop.
        return []

    # A single shared common word must not ground an answer. BM25 alone will
    # happily return a scheme document for "how do I invest money in bitcoin"
    # because both contain "money", and a high score on one term reads as
    # confident. Requiring two distinct query terms in the document is the
    # cheapest thing that separates a real topical match from a coincidence.
    #
    # ponytail: term overlap, not semantic similarity. A user who asks entirely
    # in synonyms still misses; the fix for that is embeddings, and the cost is
    # ~400MB of a budget we do not have.
    qterms = _terms(text)
    # Never fewer than min_terms, even for a one-word query. One word is small
    # talk ("what is your name") or too vague to answer from a document, and
    # letting it ground on any row that mentions the word is how the device
    # answered "what is your name" with a ration-card procedure.
    need = min_terms
    kept = []
    for r in rows:
        blob = f"{r['title']} {r['aliases']} {r['body']}".lower()
        matched = [t for t in qterms if t in blob]
        r["matched_terms"] = _distinct_words(matched, r["aliases"].lower())
        if r["matched_terms"] >= need:
            kept.append(r)

    # Rank by how many distinct query terms the document covers, and only use
    # BM25 to break ties. BM25 normalises by field length, so a document with a
    # short aliases list can outscore one that matches strictly more of the
    # question -- which is exactly the wrong answer when several documents cover
    # different facets of the same scheme ("how much", "who is eligible", "how to
    # apply"). Term coverage is the signal that routes to the right facet.
    kept.sort(key=lambda r: (-r["matched_terms"], r["score"]))
    rows = kept[:k]

    hits = [r for r in rows if r["score"] <= threshold]
    # Drop a marginal second hit rather than spend prefill tokens on it.
    if len(hits) > 1 and hits[1]["score"] > 0.5 * hits[0]["score"]:
        hits = hits[:1]
    return hits


def context_block(hits: list[dict], max_chars: int = 700) -> str:
    """Render hits as prompt context. Bounded, because this becomes LLM prefill."""
    out, used = [], 0
    for h in hits:
        line = f"- {h['title']}: {h['body']}"
        if used + len(line) > max_chars:
            break
        out.append(line)
        used += len(line)
    return "\n".join(out)


def _self_check() -> None:
    """Smallest thing that fails if retrieval breaks. No pytest, no fixtures."""
    import tempfile

    tsv = (
        "doc_id\tsection\ttitle\taliases\tbody\n"
        "pmkisan\tscheme\tPM-KISAN farmer income support\t"
        "kisan samman nidhi किसान सम्मान निधि farmer farmers money instalment\t"
        "PM-KISAN pays 6000 rupees per year to eligible landholding farmer "
        "families in three equal instalments of 2000 rupees.\n"
        "pmjay\tscheme\tAyushman Bharat health cover\t"
        "ayushman card आयुष्मान hospital cashless\t"
        "Ayushman Bharat PM-JAY gives eligible families health cover of up to "
        "5 lakh rupees per year for hospital treatment.\n"
        "ors\thealth\tORS for dehydration\t"
        "loose motion dast दस्त diarrhoea child\t"
        "For loose motions give ORS solution after every loose stool and see a "
        "doctor if the child is drowsy or passing very little urine.\n"
    )
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        tsv_path, db_path = Path(d) / "c.tsv", Path(d) / "c.db"
        tsv_path.write_text(tsv, encoding="utf-8")

        assert build(tsv_path, db_path) == 3
        con = ensure_db(tsv_path, db_path)
        try:
            # On-topic English retrieves the right row.
            q = "How much money do farmers get under PM KISAN"
            hits = search(con, q)
            assert hits and hits[0]["doc_id"] == "pmkisan", (q, _fts_query(q), hits)

            # Raw Devanagari alone retrieves it too -- this is why the app
            # appends the untranslated ASR transcript to the query.
            q = "किसान सम्मान निधि"
            hits = search(con, q)
            assert hits and hits[0]["doc_id"] == "pmkisan", ("devanagari", _fts_query(q), hits)

            # A different domain does not bleed across.
            q = "my child has loose motion what should I give"
            hits = search(con, q)
            assert hits and hits[0]["doc_id"] == "ors", (q, _fts_query(q), hits)

            # Off-topic returns nothing even with no threshold, because no term
            # matches at all. This is what makes the ungrounded path free.
            assert search(con, "who won the world cup in 1998") == []
            assert search(con, "tell me a joke please") == []

            # One shared common word is not a topical match. Without this rule
            # "invest money in bitcoin" retrieves the scheme document -- with a
            # strong BM25 score -- purely because both mention money.
            assert search(con, "how do I invest money in bitcoin") == []
            # ...and the same query grounds fine once a second term really matches.
            hits = search(con, "how much money for farmers")
            assert hits and hits[0]["doc_id"] == "pmkisan", hits
            # A single content word is not a question the corpus can answer.
            # "What is your name?" is one term, "name", and it grounded on the
            # ration-card name-correction row; a person heard the device answer
            # small talk with a scheme. Two terms, always.
            assert search(con, "what is your name") == []
            assert search(con, "ayushman") == []
            assert search(con, "ayushman card"), "two real terms must still ground"

            # A threshold stricter than this tiny corpus can score rejects
            # everything -- the calibration failure mode, asserted so nobody
            # ships a number measured on a different corpus.
            assert search(con, "How much money do farmers get under PM KISAN",
                          threshold=-50.0) == []

            # FTS5 operator injection from ASR/NMT text must not raise.
            for hostile in ['* OR NEAR(', 'x" AND "y', 'foo* bar^', '"""', 'a AND']:
                assert isinstance(search(con, hostile), list), hostile

            # Stopword-only input yields no query at all.
            assert _fts_query("what is the") is None
            assert _fts_query("") is None

            # Devanagari and Tamil words must survive tokenization intact --
            # a regex on \w silently shatters them into single consonants.
            assert _tokens("किसान सम्मान") == ["किसान", "सम्मान"]
            assert _tokens("வணக்கம்") == ["வணக்கம்"]

            # context_block never exceeds its budget.
            hits = search(con, "ayushman card hospital cover", k=2)
            assert len(context_block(hits, max_chars=40)) <= 40
            assert context_block([]) == ""
        finally:
            con.close()
    # A proper noun in two scripts is one word. "delhi" + "दिल्ली" against an
    # aliases string that carries them as a pair must count once, or every
    # question that mentions the city grounds on the Delhi row.
    assert _distinct_words(["delhi", "दिल्ली"], "ews delhi दिल्ली apply") == 1
    assert _distinct_words(["delhi", "दिल्ली", "ews"], "ews delhi दिल्ली apply") == 2
    assert _distinct_words(["train", "delhi"], "ews delhi दिल्ली apply") == 2
    assert _distinct_words(["दिल्ली", "पटना"], "ews delhi दिल्ली apply") == 2
    # Neighbours in the list are not pairs. "बिहार apply" sits in the string
    # but "apply" is its own word; only "bihar बिहार" collapses.
    assert _distinct_words(["apply", "bihar", "बिहार", "ews", "ईडब्ल्यूएस"],
                           "ews ईडब्ल्यूएस bihar बिहार apply आवेदन") == 3

    assert "tb" in _terms("where do I get TB medicine"), "TB dropped as a short token"
    assert "ka" not in _terms("kisan ka paisa"), "noise token kept"
    print("retrieval self-check OK")


if __name__ == "__main__":
    _self_check()
