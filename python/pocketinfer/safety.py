"""Things the device must handle before, or instead of, asking a language model.

Two rules, both adapted from Akansha's engine design:

  1. An emergency short-circuits everything. Chest pain, breathlessness, a fall,
     bleeding -- these are answered by a fixed script, immediately, with no model
     in the path at all. A six second wait for a 2B model to phrase something is
     the wrong behaviour when someone needs an ambulance.

  2. In high-stakes domains, a retrieval miss is a refusal, not a prompt. If the
     question is clearly about medicine, a scheme or a symptom and we retrieved
     nothing, we say so and name who to ask. The alternative -- handing the
     question to the model ungrounded -- is exactly how a device invents a dose.

The generic "I don't know" already existed; what this adds is knowing *which*
kind of not-knowing it is, so the device can name the right next step. "Ask at
the Common Service Centre" is useless advice for a fever.

Stdlib only and no hardware imports, so it is tested on any laptop.
"""
import re

# Matched against the raw transcript AND the English translation, because ASR
# may return either script and the translation may be poor precisely when the
# speaker is distressed. Substring matching on purpose: "गिर गय" catches both
# "गिर गया" and "गिर गयी".
EMERGENCY_PATTERNS = [
    # Devanagari
    "सीने में दर्द", "छाती में दर्द", "साँस नहीं", "सांस नहीं", "साँस फूल",
    "बेहोश", "गिर गय", "खून बह", "खून नहीं रुक", "दौरा", "लकवा",
    "आत्महत्या", "जान देने",
    # Romanised Hindi
    "seene me dard", "chhaati me dard", "saans nahi", "saans phool",
    "behosh", "gir gaya", "gir gayi", "khoon bah", "aatmahatya",
    # English, in case the query arrives translated
    "chest pain", "cannot breathe", "can't breathe", "not breathing",
    "unconscious", "fainted", "collapsed", "bleeding heavily", "stroke",
    "seizure", "suicide", "kill myself",
]

EMERGENCY_REPLY = {
    "hi": ("यह इंतज़ार करने वाली बात नहीं है। घर में किसी को अभी आवाज़ दीजिए "
           "और एम्बुलेंस के लिए एक सौ आठ पर फ़ोन कीजिए।"),
    "ta": ("இது காத்திருக்க வேண்டிய விஷயம் அல்ல. வீட்டில் யாரையாவது "
           "இப்போதே கூப்பிடுங்கள், ஆம்புலன்ஸுக்கு நூற்று எட்டு என்ற எண்ணை அழையுங்கள்."),
    "en": ("This cannot wait. Call someone in the house right now, and dial "
           "one-oh-eight for an ambulance."),
}

# Domains where a retrieval miss must not fall through to the model.
GATED_DOMAINS = {
    "medicine": [
        "goli", "dawai", "dawa", "dose", "khuraak", "parchi", "tablet", "syrup",
        "गोली", "दवाई", "दवा", "खुराक", "पर्ची", "इंजेक्शन", "एंटीबायोटिक", "इंसुलिन",
        "medicine", "medicines", "medication", "medications",
        "tablet", "dose", "dosage", "prescription", "injection",
        # Drug classes and the names people actually say. A question naming a
        # specific drug is the highest-risk kind there is, and it will not
        # contain the word "medicine".
        "antibiotic", "painkiller", "paracetamol", "insulin", "capsule",
        "ointment", "drops", "steroid", "antacid",
        # Including names ASR produces by mistake. Measured: "insulin" comes
        # back as "penicillin" often enough to matter, and a question the gate
        # does not recognise is a question the model answers unaided.
        "penicillin", "amoxicillin", "azithromycin", "ibuprofen", "aspirin",
        # "tab" as well as "tablet": the camera reads abbreviations off
        # packaging that nobody says out loud.
        "tab", "caps", "mg", "ml", "sachet", "strip",
        "syrup", "vaccine", "pill", "pills",
    ],
    "schemes": [
        "yojana", "scheme", "subsidy", "sarkari", "aavedan",
        "योजना", "सरकारी", "आवेदन", "सब्सिडी", "पेंशन",
        "scheme", "subsidy", "pension", "eligibility", "government",
    ],
    "health": [
        "bimari", "ilaj", "bukhar", "dard", "infection",
        "बीमारी", "इलाज", "बुखार", "दर्द", "सूजन",
        "disease", "illness", "fever", "pain", "infection", "symptom", "treatment",
    ],
}

# What to say on a gated miss. Each names a specific, reachable next step --
# a generic "visit a service centre" is useless advice for a fever.
GATED_MISS_REPLY = {
    "medicine": {
        "hi": ("इसका जवाब अपनी तरफ़ से देना मेरे लिए सही नहीं है। डॉक्टर, "
               "केमिस्ट या आशा दीदी से पूछ लीजिए।"),
        "en": ("It would not be right for me to answer that myself. Please ask "
               "a doctor, the chemist, or your ASHA worker."),
    },
    "schemes": {
        "hi": ("मुझे इस योजना की पक्की जानकारी नहीं है, और अंदाज़े से बताना ठीक नहीं। "
               "नज़दीकी जन सेवा केंद्र या पंचायत भवन में पूछिए।"),
        "en": ("I do not have reliable information about that scheme, and "
               "guessing would be wrong. Please ask at the nearest Common "
               "Service Centre or panchayat bhawan."),
    },
    "health": {
        "hi": ("इसका सही जवाब देखे बिना नहीं दिया जा सकता। आशा दीदी या नज़दीकी "
               "स्वास्थ्य केंद्र में दिखाइए, वहाँ जाँच मुफ़्त होती है। "
               "सलाह के लिए एक सौ चार पर फ़ोन कर सकते हैं।"),
        "en": ("That cannot be answered properly without an examination. Please "
               "see your ASHA worker or the nearest health centre, where checks "
               "are free. You can also call 104 for advice."),
    },
}


# Short tokens that are real medicine vocabulary and also fragments of ordinary
# words: "mg" sits inside "mgnrega", "tab" inside "table", "ml" inside "html",
# "caps" inside "capsize". Plain substring matching sends a missed MGNREGA
# question to the medicine refusal -- telling someone asking about a job card to
# go and see a doctor. These four match as whole words instead.
#
# Deliberately only these four. Word boundaries cut both ways: "goli" has to keep
# matching "goliyan", and on this gate a false negative is the dangerous
# direction. ASCII only, because \b is unreliable against Devanagari -- Python's
# \w excludes combining marks, the same defect retrieval._tokens() works around.
_WORD_ONLY = {n: re.compile(r"\b%s\b" % n) for n in ("mg", "ml", "tab", "caps")}


def _contains(text, needles):
    t = (text or "").lower()
    for n in needles:
        rx = _WORD_ONLY.get(n)
        if rx.search(t) if rx else n in t:
            return True
    return False


def is_emergency(*texts):
    """True if any of the given strings looks like a medical emergency.

    Pass both the raw transcript and the translation: ASR may return either
    script, and translation quality is worst exactly when someone is distressed.
    """
    return any(_contains(t, EMERGENCY_PATTERNS) for t in texts if t)


def emergency_reply(lang):
    return EMERGENCY_REPLY.get(lang, EMERGENCY_REPLY["en"])


def gated_domain(*texts):
    """Which high-stakes domain this question falls in, or None.

    Checked only when retrieval found nothing. Order matters: medicine before
    health, because "what dose of fever medicine" is a medicine question and
    the medicine refusal is the more specific advice.
    """
    for domain in ("medicine", "schemes", "health"):
        if any(_contains(t, GATED_DOMAINS[domain]) for t in texts if t):
            return domain
    return None


def gated_reply(domain, lang):
    by_lang = GATED_MISS_REPLY[domain]
    return by_lang.get(lang, by_lang["en"])


# Phrases that assert something is safe, permitted, or can be stopped. In a
# medicine context these are the answers that cause harm, and a 1.5B model
# produces them even when the retrieved facts say the opposite -- measured:
# asked whether an antibiotic could be stopped once the fever passed, it
# answered "yes, it is appropriate to stop", contradicting the very paragraph
# it was given. It also called an invented drug ("Kansi", an ASR corruption of
# "kaun si") safe in pregnancy.
#
# So the model's word is not trusted on this. If a medical question comes back
# with an assertion of safety, the assertion is replaced by a refusal. Blunt,
# and it will occasionally discard a correct answer -- which is the right
# direction to be wrong in.
ASSERTS_SAFE = re.compile(
    r"("
    r"is (generally |usually |quite |perfectly )?safe"
    r"|are (generally |usually )?safe"
    r"|it is (safe|fine|okay|ok|appropriate|advisable)"
    r"|you (can|may|should) (take|stop|use|discontinue)"
    r"|(is|are) (fine|okay|ok) to (take|stop|use)"
    r"|no need to (continue|worry)"
    r"|yes[, ].{0,40}(stop|discontinue|safe)"
    # The inverse is as harmful: telling someone not to take a treatment.
    # 14 Sept: "should I stop TB medicine" lost a letter in speech
    # recognition, became "should I take TB medication" in translation, and
    # the model answered "you should not take TB medicine". "Do not stop"
    # stays allowed; only not taking or not continuing is caught.
    r"|(should|must) not (take|continue|use)"
    r"|shouldn.t (take|continue|use)"
    r"|(no need|not necessary|don.t need|do not need) to (take|continue)"
    r")", re.I)


def asserts_safety(text):
    """True if the answer vouches for a medicine, dose, or stopping treatment."""
    return bool(ASSERTS_SAFE.search(text or ""))


# The model saying it does not know is correct behaviour. Sending a fever to a
# Common Service Centre is not -- that is where you go for a form, not a
# symptom. When a medical question ends in a non-answer we keep the refusal and
# replace only the next step with one that fits the domain.
SAYS_UNKNOWN = re.compile(
    r"(do not know|don.t know|not sure|no information|cannot say|can.t say"
    r"|do not have (that |this |any )?information)", re.I)


def says_unknown(text):
    """True if the answer is a non-answer, whatever wording the model chose."""
    return bool(SAYS_UNKNOWN.search(text or ""))


def _self_check():
    # Emergencies are caught in either script and in translation.
    assert is_emergency("मेरे सीने में दर्द है")
    assert is_emergency("", "I have chest pain and cannot breathe")
    assert is_emergency("dadi gir gayi hai")
    assert is_emergency("वह बेहोश हो गया")
    # Ordinary questions are not emergencies. This matters more than the
    # positives: a false positive replaces a real answer with an ambulance.
    assert not is_emergency("किसान सम्मान निधि में कितने पैसे मिलते हैं")
    assert not is_emergency("How much money do farmers get")
    assert not is_emergency("बच्चे को दस्त लग गए हैं")
    assert not is_emergency("")
    assert not is_emergency(None)

    # Gated domains route to the right adviser.
    assert gated_domain("बुखार में कितनी गोली दूं") == "medicine"
    assert gated_domain("which antibiotic should I take") == "medicine"
    assert gated_domain("इस योजना के लिए कौन पात्र है") == "schemes"
    assert gated_domain("what is the eligibility for this scheme") == "schemes"
    assert gated_domain("मुझे बुखार है") == "health"
    # Out of scope stays out of scope -- it must reach the ordinary
    # "I don't know" path, not a medical refusal.
    assert gated_domain("who won the world cup in 1998") is None

    # "mg" is medicine vocabulary and also the first two letters of MGNREGA.
    # Before word boundaries a missed job-card question was answered "see a
    # doctor". Retrieval normally catches these; this is the miss path.
    assert gated_domain("how do I get an mgnrega job card") != "medicine"
    assert gated_domain("what is the mgnrega wage per day") != "medicine"
    assert gated_domain("there is a table in the room") is None
    # ...and real drug vocabulary still gates.
    assert gated_domain("is 500 mg too much for a child") == "medicine"
    assert gated_domain("how many ml of syrup") == "medicine"
    assert gated_domain("one tab twice a day") == "medicine"
    # Inflected romanised forms must keep matching -- these stay substring.
    assert gated_domain("kitni goliyan leni hain") == "medicine"
    assert gated_domain("कल मौसम कैसा रहेगा") is None

    # Every reply exists in every language we claim.
    for lang in ("hi", "ta", "en"):
        assert emergency_reply(lang)
    for domain in GATED_DOMAINS:
        for lang in ("hi", "en"):
            assert gated_reply(domain, lang)

    # The two failures measured on real recordings.
    assert asserts_safety("Yes, if the fever has subsided, it is appropriate to stop the antibiotic")
    assert asserts_safety("Kansi medicine is generally safe for pain management")
    assert asserts_safety("You can take two tablets after food")
    # ...and answers that must NOT be flagged, because they are already correct.
    assert not asserts_safety("Please ask a doctor or your ASHA worker.")
    assert not asserts_safety("PM-KISAN pays 6000 rupees per year in three instalments.")
    assert not asserts_safety("Blood pressure medicine should not be stopped without medical advice.")
    assert not asserts_safety("")

    assert says_unknown("I do not know how many units were taken.")
    assert says_unknown("I don't know the weather forecast.")
    assert says_unknown("I do not have that information.")
    assert not says_unknown("PM-KISAN pays 6000 rupees per year.")
    assert not says_unknown("")

    print("safety self-check OK")


if __name__ == "__main__":
    _self_check()
