"""Regex patterns and rule-based KIE baseline for receipts (no ML deps)."""
from __future__ import annotations

import re

_NUM_DATE = r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
           r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_WORD_DATE = rf"\b\d{{1,2}}[\s/\-\.]{_MONTHS}[\s/\-\.]\d{{2,4}}\b|\b{_MONTHS}[\s/\-\.]\d{{1,2}}[,\s/\-\.]+\d{{2,4}}\b"
DATE_RE = re.compile(f"(?:{_NUM_DATE})|(?:{_WORD_DATE})", re.IGNORECASE)
MONEY_RE = re.compile(r"(?:RM|USD|SGD|MYR|\$)?\s*\d{1,3}(?:,\d{3})*\.\d{2}\b", re.IGNORECASE)
_MONEY_OCR_SPAN = re.compile(r"[\dOolI][\dOolI.,]*[\dOolI]")

def repair_money_ocr(s: str) -> str:
    """Fix TrOCR OCR errors: O/o→0, l/I→1, European comma→dot."""
    def _fix(m: re.Match[str]) -> str:
        t = m.group(0).translate(str.maketrans("Ool", "001")).replace("I", "1")
        if t.count(",") == 1 and "." not in t:
            t = t.replace(",", ".")
        return t
    return _MONEY_OCR_SPAN.sub(_fix, s)

_DATE_RE, _MONEY_RE = DATE_RE, MONEY_RE
_TOTAL_NEGATIVE = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|sub|round(?:ing|ed)?|change|cash\s+tendered|tendered|"
    r"balance|credit|debit|card|visa|master(?:card)?|paid|payment|kembalian|discount|"
    r"service|charge|tax\s+(?:only|\d)|gst\s+\d|sst\s+\d|qty|item|no\.)\b", re.IGNORECASE)
_TOTAL_STRONG = re.compile(
    r"\b(grand\s*total|amount\s*(?:due|payable)|nett?\s*total|total\s*(?:due|amt|amount))\b", re.IGNORECASE)
_TOTAL_WEAK = re.compile(r"\btotal\b|\bamount\b", re.IGNORECASE)
_HEADER_JUNK = re.compile(
    r"^\s*(tax\s*invoice|invoice|receipt|cash\s*(?:sale|bill)|bill|original(?:\s*copy)?|"
    r"copy|reprint|duplicate|welcome|thank\s*you|customer\s*copy|merchant\s*copy)\s*[:\-]?\s*$", re.IGNORECASE)
_ADDR_EXCLUDE = re.compile(
    r"\b(tel(?:ephone|\.?)?|phone|fax|h\/?p|hp|mobile|email|e-mail|gst|sst|"
    r"reg(?:istration)?(?:\s*no\.?)?|co(?:mpany)?\s*no\.?|kad|vat|tin|www\.|http|\.com|\.my)\b", re.IGNORECASE)
_ADDR_ANCHOR = re.compile(
    r"\b(NO\.?|LOT|JALAN|JLN|TAMAN|TMN|BANDAR|BDR|PLAZA|GROUND|GRD|FLR|FLOOR|KAWASAN|SEKSYEN|BLOCK|BLK|"
    r"MALL|LORONG|LRG|PERSIARAN|PUSAT|DESA|PARADIGM|AEON|CITTA|SQUARE|CENTRE|CENTER|TINGKAT|MILES|"
    r"BUILDING|BLDG|UTAMA)\b|\b\d{5}\b", re.IGNORECASE)
_POSTCODE_RE = re.compile(r"\b\d{5}\b")
_ADDR_CONTINUATION = re.compile(
    r"\b(SELANGOR|JOHOR|KEDAH|KELANTAN|MELAKA|MALACCA|PAHANG|PERAK|PERLIS|PENANG|PULAU\s+PINANG|SABAH|"
    r"SARAWAK|TERENGGANU|KUALA\s+LUMPUR|KL|PUTRAJAYA|LABUAN|MALAYSIA|DARUL\s+EHSAN|DARUL\s+KHUSUS|"
    r"DARUL\s+MAKMUR|DARUL\s+NAIM|D\.E\.?|N\.S\.?|CHERAS|PUCHONG|SUBANG|KLANG|SHAH\s+ALAM|KAJANG|KEPONG|"
    r"PETALING|SKUDAI|JAYA|BRINCHANG|BALAKONG|DENGKIL|SERDANG|SETIA\s+ALAM|SETAPAK|BATANG\s+BERJUNTAI|"
    r"AMPANG|GOMBAK|JOHOR\s+BAHRU|SEREMBAN|IPOH|KUANTAN|MASAI|BAHRU)\b", re.IGNORECASE)
_ADDR_TERMINATOR = re.compile(
    r"\b(INVOICE(?:\s+NO)?|INV\s+NO|TAX\s+INVOICE|CASH(?:IER|\s+SALES?|\s+RECEIPT)|BILL\s+(?:TO|NO)|"
    r"RECEIPT\s+NO|TABLE\s+NO?\b|TABLE\s+\d|COUNTER|GUEST\s+CHECK|ORDER\s+NO|DOC\s*#|"
    r"SIMPLIFIED(?:\s+TAX)?|SHOPPING\s+HOURS|ADJUSTMENT\s+NOTE|PAY\s+BY|CARRY\s+OUT|WEBSITE|\bBRN\b|"
    r"SITE\s+\d|POSTED|RETAIL\b|TAKEAWAY|OWNED\s+BY|SUN-THU|MON-SUN|ROC\s+NO|DEPT\s+(?:DOC|SO)|"
    r"TEL(?:EPHONE)?\s*(?:NO|[:.])|PHONE\s*(?:NO|[:.])|FAX\s*(?:NO|[:.]))\b", re.IGNORECASE)
_COMPANY_TOKEN = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BERHAD|ENTERPRISE|HOLDINGS|TRADING|MARKETING|CORPORATION|CORP\.?|"
    r"CO\.?\s*M\s*BHD|CO\.\s*LTD\.?|LIMITED|INC\.?)\b", re.IGNORECASE)
_ADDR_LEADING_JUNK_RE = re.compile(
    r"^\s*(?:CO\.?\s*NO\.?\s*[\w\-]+|COMPANY\s*NO\.?\s*[\w\-]+|REG(?:ISTRATION)?\s*NO\.?\s*[\w\-]+|"
    r"GST(?:\s*NO\.?)?\s*[\w\-]+|SST(?:\s*NO\.?)?\s*[\w\-]+|TIN(?:\s*NO\.?)?\s*[\w\-]+)[,\s:;\-]*", re.IGNORECASE)

def rule_based_assign(region_texts: list[str], bbox_list: list[list[float]]) -> dict[str, str]:
    """Assign company/date/address/total using spatial + regex heuristics."""
    from models.rule_fields import _pick_address, _pick_company, extract_date, extract_total
    assigned: dict[str, str] = {}
    used: set[int] = set()
    date_pick = extract_date(region_texts)
    if date_pick is not None:
        assigned["date"] = date_pick[1]
        used.add(date_pick[0])
    total_pick = extract_total(region_texts, bbox_list)
    if total_pick is not None:
        assigned["total"] = total_pick[1]
        used.add(total_pick[0])
    company_pick = _pick_company(region_texts, bbox_list, used)
    company_y = 0.0
    if company_pick is not None:
        assigned["company"] = company_pick[1]
        used.add(company_pick[0])
        company_y = bbox_list[company_pick[0]][1] if company_pick[0] < len(bbox_list) else 0.0
    total_y = bbox_list[total_pick[0]][1] if total_pick is not None and total_pick[0] < len(bbox_list) else 0.0
    date_y = bbox_list[date_pick[0]][1] if date_pick is not None and date_pick[0] < len(bbox_list) else 0.0
    addr = _pick_address(region_texts, bbox_list, used, company_y, total_y, date_y)
    if addr:
        assigned["address"] = addr
    return assigned

__all__ = ["DATE_RE", "MONEY_RE", "rule_based_assign", "repair_money_ocr"]
