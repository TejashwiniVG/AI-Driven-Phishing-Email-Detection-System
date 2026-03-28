import re, socket
from typing import Dict, List, Tuple
import tldextract
import dns.resolver
import validators
import idna

try:
    import dkim as dkimpy
except Exception:
    dkimpy = None

def parse_headers(headers_text: str) -> Dict[str, str]:
    """
    Very light parser: returns a dict of header-name -> value (last occurrence wins).
    Assumes headers_text is the raw header block (not full message with body).
    """
    if not isinstance(headers_text, str) or not headers_text.strip():
        return {}
    lines = headers_text.splitlines()
    # unfold continuation lines
    unfolded = []
    for line in lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += " " + line.strip()
        else:
            unfolded.append(line)
    headers = {}
    for line in unfolded:
        if ":" in line:
            name, val = line.split(":", 1)
            headers[name.strip().lower()] = val.strip()
    return headers

def _extract_domain_from_addr(addr: str) -> str:
    if not addr:
        return ""
    m = re.search(r'@([A-Za-z0-9.\-\_]+)', addr)
    return m.group(1).lower() if m else ""

def has_spf_record(domain: str) -> bool:
    if not domain:
        return False
    try:
        answers = dns.resolver.resolve(domain, 'TXT', lifetime=2.0)
        for rdata in answers:
            txt = "".join([b.decode('utf-8', 'ignore') for b in rdata.strings])
            if txt.lower().startswith("v=spf1"):
                return True
    except Exception:
        return False
    return False

def has_dmarc_policy(domain: str) -> Tuple[bool, str]:
    if not domain:
        return (False, "")
    try:
        q = f"_dmarc.{domain}"
        answers = dns.resolver.resolve(q, 'TXT', lifetime=2.0)
        for rdata in answers:
            txt = "".join([b.decode('utf-8', 'ignore') for b in rdata.strings])
            if "v=DMARC1" in txt.upper():
                m = re.search(r'\bp=([a-zA-Z]+)', txt)
                pol = m.group(1).lower() if m else ""
                return (True, pol)
    except Exception:
        return (False, "")
    return (False, "")

def has_dkim_signature(headers: Dict[str,str]) -> bool:
    if not headers:
        return False
    return 'dkim-signature' in headers

URL_RE = re.compile(r'(https?://[^\s)>\"]+)', re.IGNORECASE)

SUSPICIOUS_TLDS = set("zip mov click country work link xyz top gq cf ml tk men cam win loan stream quest mom".split())

def extract_urls(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    return URL_RE.findall(text)

def domain_features(url: str, claimed_domain: str="") -> Dict[str, object]:
    feats = {"url": url, "domain": "", "subdomain": "", "suffix": "", "issues": []}
    if not validators.url(url):
        feats["issues"].append("invalid_url")
        return feats
    parts = tldextract.extract(url)
    feats["domain"] = parts.registered_domain
    feats["subdomain"] = parts.subdomain
    feats["suffix"] = parts.suffix

    dots = url.count(".")
    at_sign = url.count("@")
    hyphens = url.count("-")
    digits = sum(c.isdigit() for c in url)
    if dots >= 5: feats["issues"].append("too_many_dots")
    if at_sign > 0: feats["issues"].append("contains_at")
    if hyphens >= 4: feats["issues"].append("many_hyphens")
    if digits >= 10: feats["issues"].append("many_digits")

    if re.search(r'https?://\d{1,3}(\.\d{1,3}){3}', url):
        feats["issues"].append("ip_address_url")

    if "xn--" in url:
        feats["issues"].append("punycode_domain")

    if parts.suffix.lower() in SUSPICIOUS_TLDS:
        feats["issues"].append(f"suspicious_tld:{parts.suffix.lower()}")

    if claimed_domain:
        try:
            claimed = tldextract.extract(claimed_domain).registered_domain.lower()
            actual = feats["domain"].lower()
            if claimed and actual and claimed != actual:
                feats["issues"].append(f"mismatch:{claimed}->{actual}")
        except Exception:
            pass

    return feats

def risk_from_features(url_feats: Dict[str, object]) -> int:
    score = 0
    for issue in url_feats.get("issues", []):
        if issue.startswith("suspicious_tld"): score += 20
        elif issue == "ip_address_url": score += 25
        elif issue == "punycode_domain": score += 15
        elif issue.startswith("mismatch"): score += 20
        elif issue == "too_many_dots": score += 10
        elif issue == "contains_at": score += 10
        elif issue == "many_hyphens": score += 10
        elif issue == "many_digits": score += 10
        elif issue == "invalid_url": score += 15
    return max(0, min(100, score))

def compute_authenticity(headers_text: str) -> Dict[str, object]:
    headers = parse_headers(headers_text or "")
    from_domain = _extract_domain_from_addr(headers.get("from", ""))
    reply_domain = _extract_domain_from_addr(headers.get("reply-to", ""))

    spf = has_spf_record(from_domain) if from_domain else False
    dmarc_exists, dmarc_p = has_dmarc_policy(from_domain) if from_domain else (False, "")
    dkim_present = has_dkim_signature(headers)

    risk = 0
    if not spf: risk += 25
    if not dkim_present: risk += 25
    if not dmarc_exists: risk += 25
    if dmarc_exists and dmarc_p in ("none", ""): risk += 10
    if reply_domain and from_domain and reply_domain != from_domain:
        risk += 15

    return {
        "from_domain": from_domain,
        "reply_to_domain": reply_domain,
        "spf_published": spf,
        "dmarc_present": dmarc_exists,
        "dmarc_policy": dmarc_p,
        "dkim_present": dkim_present,
        "auth_risk": max(0, min(100, risk))
    }

def analyze_urls_in_text(text: str, claimed_domain: str="") -> Dict[str, object]:
    urls = extract_urls(text or "")
    feats = [domain_features(u, claimed_domain) for u in urls]
    per_url = [{"url": f["url"], "domain": f["domain"], "issues": f["issues"], "risk": risk_from_features(f)} for f in feats]
    overall = max([p["risk"] for p in per_url], default=0)
    return {"urls": per_url, "url_risk": overall}
