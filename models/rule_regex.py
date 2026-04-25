"""Regex patterns for the rule-based KIE baseline (no ML dependencies).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: DATE_RE and MONEY_RE are the primary field detectors; _TOTAL_NEGATIVE,
    _TOTAL_STRONG, _TOTAL_WEAK drive the keyword ranking in extract_total.
"""
from __future__ import annotations

import re

# DATE_RE accepts:
#   2024/08/01, 01-08-2024, 01.08.2019, 1-8-24   (numeric, 3-part)
#   01-AUG-2019, 1 AUG 2019, 1/AUG/2019          (word month between digits)
#   August 1, 2019, AUG 01 2019                  (word month first)
_NUM_DATE = r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_WORD_DATE = (
    rf"\b\d{{1,2}}[\s/\-\.]{_MONTHS}[\s/\-\.]\d{{2,4}}\b"
    rf"|\b{_MONTHS}[\s/\-\.]\d{{1,2}}[,\s/\-\.]+\d{{2,4}}\b"
)
DATE_RE = re.compile(f"(?:{_NUM_DATE})|(?:{_WORD_DATE})", re.IGNORECASE)

# Money — matches ``12.30``, ``$12.30``, ``RM12.30``, ``1,234.56``.
MONEY_RE = re.compile(r"(?:RM|USD|SGD|MYR|\$)?\s*\d{1,3}(?:,\d{3})*\.\d{2}\b", re.IGNORECASE)

# OCR-confused digit spans: a numeric-looking run where TrOCR may have
# substituted ``O``/``o`` for ``0``, ``l``/``I`` for ``1``, or the European
# ``,`` for the decimal ``.``.  The span must start and end with a digit or
# an OCR-confused digit so we don't pick up pure-letter tokens like ``IO``.
_MONEY_OCR_SPAN = re.compile(r"[\dOolI][\dOolI.,]*[\dOolI]")


def repair_money_ocr(s: str) -> str:
    """Fix common TrOCR money OCR errors inside digit-only spans.

    Substitutes ``O``/``o``→``0`` and ``l``/``I``→``1`` inside spans that
    already look numeric, and converts a lone ``,`` decimal separator to
    ``.`` (European → US format) so ``MONEY_RE`` can match ``43,50`` or
    ``43.5O`` the same way it matches ``43.50``.  Non-numeric tokens are
    left untouched because the span anchors require a digit-like boundary.
    """
    def _fix(m: re.Match[str]) -> str:
        t = m.group(0).translate(str.maketrans("Ool", "001")).replace("I", "1")
        # The ``,``→``.`` swap handles the European-format case
        # (single comma, no existing dot) so ``43,50`` parses as ``43.50``.
        if t.count(",") == 1 and "." not in t:
            t = t.replace(",", ".")
        return t
    return _MONEY_OCR_SPAN.sub(_fix, s)


# Backwards-compatible aliases.
_DATE_RE = DATE_RE
_MONEY_RE = MONEY_RE

# Words that disqualify a money region from being TOTAL.
_TOTAL_NEGATIVE = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|sub|round(?:ing|ed)?|"
    r"change|cash\s+tendered|tendered|balance|credit|debit|"
    r"card|visa|master(?:card)?|paid|payment|kembalian|"
    r"discount|service|charge|tax\s+(?:only|\d)|gst\s+\d|sst\s+\d|"
    r"qty|item|no\.)\b",
    re.IGNORECASE,
)

# Positive TOTAL cues — strongest signal is "GRAND TOTAL" > "TOTAL" > "AMOUNT".
_TOTAL_STRONG = re.compile(
    r"\b(grand\s*total|amount\s*(?:due|payable)|nett?\s*total|total\s*(?:due|amt|amount))\b",
    re.IGNORECASE,
)
_TOTAL_WEAK = re.compile(r"\btotal\b|\bamount\b", re.IGNORECASE)

# Header junk the top-line heuristic should skip when picking a company.
_HEADER_JUNK = re.compile(
    r"^\s*(tax\s*invoice|invoice|receipt|cash\s*(?:sale|bill)|"
    r"bill|original(?:\s*copy)?|copy|reprint|duplicate|"
    r"welcome|thank\s*you|customer\s*copy|merchant\s*copy)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Phone / tax ID / reg-no patterns that often sneak into the address block.
_ADDR_EXCLUDE = re.compile(
    r"\b(tel(?:ephone|\.?)?|phone|fax|h\/?p|hp|mobile|email|e-mail|"
    r"gst|sst|reg(?:istration)?(?:\s*no\.?)?|co(?:mpany)?\s*no\.?|"
    r"kad|vat|tin|www\.|http|\.com|\.my)\b",
    re.IGNORECASE,
)

# Address-block anchors: Malaysian-receipt address-line openers (street /
# floor / mall tokens) plus the 5-digit postcode.  The topmost line that
# matches is the first line of the postal address regardless of where
# attention fell.  Postcode alternative lets the search terminate on the
# tail line for receipts whose only address-like cue is a postcode +
# city/state line; a backward-extend then recovers the street prefix.
_ADDR_ANCHOR = re.compile(
    r"\b(NO\.?|LOT|JALAN|JLN|TAMAN|TMN|BANDAR|BDR|PLAZA|"
    r"GROUND|GRD|FLR|FLOOR|KAWASAN|SEKSYEN|BLOCK|BLK|MALL|"
    r"LORONG|LRG|PERSIARAN|PUSAT|DESA|PARADIGM|AEON|CITTA|"
    r"SQUARE|CENTRE|CENTER|TINGKAT|MILES|BUILDING|BLDG|UTAMA)\b"
    r"|\b\d{5}\b",
    re.IGNORECASE,
)
# Malaysian postcode — definitive signal of a complete postal address.
_POSTCODE_RE = re.compile(r"\b\d{5}\b")
# City/state tokens that confirm a line is still inside the address span
# even when it lacks a street keyword (e.g. SETIA ALAM continuation,
# SELANGOR tail).  Used both for same-span extension and for picking
# between (learned, rule, span) in :func:`_refine_address`.
_ADDR_CONTINUATION = re.compile(
    r"\b(SELANGOR|JOHOR|KEDAH|KELANTAN|MELAKA|MALACCA|"
    r"PAHANG|PERAK|PERLIS|PENANG|PULAU\s+PINANG|SABAH|SARAWAK|"
    r"TERENGGANU|KUALA\s+LUMPUR|KL|PUTRAJAYA|LABUAN|MALAYSIA|"
    r"DARUL\s+EHSAN|DARUL\s+KHUSUS|DARUL\s+MAKMUR|DARUL\s+NAIM|"
    r"D\.E\.?|N\.S\.?|"
    r"CHERAS|PUCHONG|SUBANG|KLANG|SHAH\s+ALAM|KAJANG|KEPONG|"
    r"PETALING|SKUDAI|JAYA|BRINCHANG|BALAKONG|DENGKIL|SERDANG|"
    r"SETIA\s+ALAM|SETAPAK|BATANG\s+BERJUNTAI|AMPANG|GOMBAK|"
    r"JOHOR\s+BAHRU|SEREMBAN|IPOH|KUANTAN|MASAI|BAHRU)\b",
    re.IGNORECASE,
)
# Keywords that mark a transition OUT of the address block into invoice
# metadata, cashier info, or post-address footer junk.  Word-boundary
# anchored so a city like TABLETON could never match TABLE.
_ADDR_TERMINATOR = re.compile(
    r"\b(INVOICE(?:\s+NO)?|INV\s+NO|TAX\s+INVOICE|"
    r"CASH(?:IER|\s+SALES?|\s+RECEIPT)|"
    r"BILL\s+(?:TO|NO)|"
    r"RECEIPT\s+NO|TABLE\s+NO?\b|TABLE\s+\d|"
    r"COUNTER|GUEST\s+CHECK|ORDER\s+NO|DOC\s*#|"
    r"SIMPLIFIED(?:\s+TAX)?|SHOPPING\s+HOURS|"
    r"ADJUSTMENT\s+NOTE|PAY\s+BY|CARRY\s+OUT|"
    r"WEBSITE|\bBRN\b|SITE\s+\d|POSTED|RETAIL\b|TAKEAWAY|"
    r"OWNED\s+BY|SUN-THU|MON-SUN|ROC\s+NO|DEPT\s+(?:DOC|SO)|"
    r"TEL(?:EPHONE)?\s*(?:NO|[:.])|"
    r"PHONE\s*(?:NO|[:.])|FAX\s*(?:NO|[:.]))\b",
    re.IGNORECASE,
)
# Company-identifier tokens — any line carrying one of these is a company
# header, never an address line.  Kept strictly to tokens that *only*
# appear in company names so we never boundary-stop on a legitimate
# address like ``LOT 1851-A``.
_COMPANY_TOKEN = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|HOLDINGS|"
    r"TRADING|MARKETING|CORPORATION|CORP\.?|"
    r"CO\.?\s*M\s*BHD|CO\.\s*LTD\.?|LIMITED|INC\.?)\b",
    re.IGNORECASE,
)
# Leading company-registration / tax-ID tokens that OCR sometimes fuses
# onto the learned address pick.  Matches only at start of string so we
# never strip a legitimate address containing NO. / LOT.  ``_ADDR_EXCLUDE``
# doesn't tolerate a period between CO and NO, hence this localised guard.
_ADDR_LEADING_JUNK_RE = re.compile(
    r"^\s*(?:CO\.?\s*NO\.?\s*[\w\-]+"
    r"|COMPANY\s*NO\.?\s*[\w\-]+"
    r"|REG(?:ISTRATION)?\s*NO\.?\s*[\w\-]+"
    r"|GST(?:\s*NO\.?)?\s*[\w\-]+"
    r"|SST(?:\s*NO\.?)?\s*[\w\-]+"
    r"|TIN(?:\s*NO\.?)?\s*[\w\-]+)[,\s:;\-]*",
    re.IGNORECASE,
)
