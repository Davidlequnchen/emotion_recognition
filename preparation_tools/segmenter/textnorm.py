"""Alignment copy of a transcript (doc §2.3 steps 2-3).

The original text is never changed. Each whitespace-separated token becomes one Word that keeps its exact
character span in the original text, plus the letter sequences ("units") the English aligner needs:
upper-case A-Z and the apostrophe, numbers and symbols written as words. Tokens that cannot be spelled
reliably (redactions such as [NAME] or ***, long digit strings, letter-digit codes) get one WILDCARD unit,
which matches any speech during alignment (as WhisperX does).
"""
from dataclasses import dataclass, field
import re
import unicodedata

WILDCARD = "*"

ONES = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN",
        "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN"]
TENS = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
ORDINALS = {"ONE": "FIRST", "TWO": "SECOND", "THREE": "THIRD", "FIVE": "FIFTH", "EIGHT": "EIGHTH",
            "NINE": "NINTH", "TWELVE": "TWELFTH"}
# A full stop after these is not a sentence end.
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "approx", "inc", "ltd", "co"}
SENTENCE_END = set(".?!…")
CLAUSE_END = set(",;:—–")
PUNCTUATION = SENTENCE_END | CLAUSE_END


@dataclass
class Word:
    text: str                 # the original token(s), exactly as written
    char_start: int           # span in the original transcript
    char_end: int
    units: list = field(default_factory=list)   # letter strings and/or WILDCARD
    punct: float = 0.0        # text cue q after this word: 1 sentence end, 0.5 clause end, 0 otherwise
    start: float = None       # filled by forced alignment (seconds in the turn)
    end: float = None
    confidence: float = None  # mean letter score

    @property
    def wildcard(self):
        return WILDCARD in self.units


def number_words(n):
    """Cardinal number as words, e.g. 1250 -> ONE THOUSAND TWO HUNDRED FIFTY."""
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [ONES[n // 100], "HUNDRED"] + (number_words(n % 100) if n % 100 else [])
    for value, name in ((10 ** 9, "BILLION"), (10 ** 6, "MILLION"), (1000, "THOUSAND")):
        if n >= value:
            return number_words(n // value) + [name] + (number_words(n % value) if n % value else [])


def ordinal_words(n):
    words = number_words(n)
    last = words[-1]
    words[-1] = ORDINALS.get(last) or (last[:-1] + "IETH" if last.endswith("Y") else last + "TH")
    return words


def year_words(n):
    """1985 -> NINETEEN EIGHTY FIVE, 2005 -> TWO THOUSAND FIVE, 2024 -> TWENTY TWENTY FOUR."""
    if 2000 <= n <= 2009:
        return number_words(n)
    high, low = divmod(n, 100)
    if low == 0:
        return number_words(high) + ["HUNDRED"]
    return number_words(high) + (["OH", ONES[low]] if low < 10 else number_words(low))


def integer_words(digits):
    """Plain or comma-grouped integer; None when the reading is ambiguous (IDs, phone numbers, codes)."""
    if re.fullmatch(r"\d{1,3}(,\d{3})+", digits):
        return number_words(int(digits.replace(",", "")))
    if not digits.isdigit() or (len(digits) > 1 and digits[0] == "0"):
        return None                                   # leading zero: an ID or code, not an amount
    if len(digits) <= 3:
        return number_words(int(digits))
    if len(digits) == 4 and (1100 <= int(digits) <= 2099):
        return year_words(int(digits))                # years and "fifteen hundred"-style amounts
    return None


def decimal_words(value):
    """3.5 -> THREE POINT FIVE (digits after the point are read one by one)."""
    whole, _, fraction = value.partition(".")
    head = integer_words(whole)
    if head is None or not fraction.isdigit():
        return None
    return head + ["POINT"] + [ONES[int(d)] for d in fraction]


def spell_number(core):
    """Numbers, money, percentages, ordinals and clock times; None if core is not one of these forms."""
    if core == "24/7":
        return ["TWENTY", "FOUR", "SEVEN"]
    if m := re.fullmatch(r"\$(\d[\d,]*)(?:\.(\d\d))?", core):
        dollars = integer_words(m[1])
        if dollars is None:
            return None
        words = dollars + (["DOLLAR"] if dollars == ["ONE"] else ["DOLLARS"])
        if m[2] and int(m[2]):
            words += number_words(int(m[2])) + (["CENT"] if int(m[2]) == 1 else ["CENTS"])
        return words
    if m := re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)%", core):
        amount = decimal_words(m[1]) if "." in m[1] else integer_words(m[1])
        return None if amount is None else amount + ["PERCENT"]
    if m := re.fullmatch(r"(\d{1,3})(st|nd|rd|th)", core, re.IGNORECASE):
        return ordinal_words(int(m[1]))
    if m := re.fullmatch(r"(\d{1,2}):(\d\d)", core):
        hour, minute = int(m[1]), int(m[2])
        if hour > 23 or minute > 59:
            return None
        return number_words(hour) + ([] if minute == 0 else ["OH", ONES[minute]] if minute < 10
                                     else number_words(minute))
    if re.fullmatch(r"\d+\.\d+", core):
        return decimal_words(core)
    if re.fullmatch(r"[\d,]+", core):
        return integer_words(core)
    return None


def letters_only(part):
    """Upper-case A-Z and apostrophe; accents removed (é -> E)."""
    part = unicodedata.normalize("NFKD", part.replace("’", "'").replace("‘", "'"))
    part = "".join(ch for ch in part if not unicodedata.combining(ch)).upper()
    return re.sub(r"[^A-Z']", "", part).strip("'")


def spell(core):
    """Units for one token core (no surrounding punctuation)."""
    if re.fullmatch(r"[\[<{(].*[\]>})]", core) or re.fullmatch(r"\*+|#+|[xX]{3,}", core):
        return [WILDCARD]                             # redaction or annotation placeholder
    number = spell_number(core)
    if number is not None:
        return number
    if re.search(r"\d", core) and re.search(r"[-/+–−]", core):
        return [WILDCARD]                             # -50, 1/2, 10-15, 555-1234: several readings, none reliable
    units = []
    for part in re.split(r"[-/]", core.replace("&", "-AND-")):
        if not part:
            continue
        if re.search(r"\d", part):
            number = spell_number(part)
            if number is None:
                return [WILDCARD]                     # codes like A12B, long digit strings
            units += number
        elif letters := letters_only(part):
            units.append(letters)
    return units or [WILDCARD]


def text_cue(trailing, core):
    """q value after a word from its trailing punctuation."""
    if set(trailing) & {"?", "!"}:
        return 1.0
    if set(trailing) & {".", "…"}:
        if trailing.startswith(".") and (core.lower() in ABBREVIATIONS or "." in core):
            return 0.0                                # "Mr." or "U.S." is not a sentence end
        return 1.0
    return 0.5 if set(trailing) & CLAUSE_END else 0.0


# A bracketed group ([background noise], <unk>, {laughs}, (inaudible speech)) is one token even if it contains
# spaces: annotations are not spoken words and must not be aligned as such.
TOKEN = re.compile(r"\[[^\]]*\]\S*|<[^>]*>\S*|\{[^}]*\}\S*|\([^)]*\)\S*|\S+")
PLACEHOLDER = re.compile(r"([\[<{(].*[\]>})])([.,!?;:…]*)")


def split_token(token):
    """(core, trailing punctuation) of one token; quotes and brackets around a word are dropped."""
    if m := PLACEHOLDER.fullmatch(token):
        return m[1], m[2]
    lead = re.match(r"^[\"'“‘(«¿¡]*", token).group()
    trail = re.search(r"[.,!?;:…—–\"'”’)»]*$", token[len(lead):]).group()
    return token[len(lead):len(token) - len(trail)], trail


def tokenize(transcript):
    """Split the transcript into Words with alignment units and text cues."""
    words = []
    pending_start = None                              # punctuation-only tokens before the first word
    for m in TOKEN.finditer(transcript):
        core, trail = split_token(m.group())
        if not re.search(r"\w|[$%&*#\[\]<>{}]", core):
            # punctuation only ("-", "...", "—"): attach to the previous word, or to the next one at the start
            token = m.group()
            cue = 1.0 if set(token) & SENTENCE_END else 0.5 if set(token) & (CLAUSE_END | {"-"}) else 0.0
            if words:
                words[-1].char_end = m.end()
                words[-1].text = transcript[words[-1].char_start:m.end()]
                words[-1].punct = max(words[-1].punct, cue)
            elif pending_start is None:
                pending_start = m.start()
            continue
        start = m.start() if pending_start is None else pending_start
        pending_start = None
        words.append(Word(text=transcript[start:m.end()], char_start=start, char_end=m.end(),
                          units=spell(core), punct=text_cue(trail, core)))
    return words


def has_punctuation(transcript):
    """False when the transcript carries no sentence or clause punctuation at all (then q gets weight 0)."""
    return any(ch in PUNCTUATION for ch in transcript)
