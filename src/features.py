import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "bank", "free",
    "confirm", "password", "signin", "webscr", "bonus", "unlock"
]

def has_ip_address(netloc: str) -> int:
    return 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", netloc) else 0

def extract_features(url: str) -> dict:
    url = (url or "").strip()
    parsed = urlparse(url if "://" in url else "http://" + url)

    hostname = parsed.netloc.lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full = f"{hostname}{path}?{query}"

    feats = {}
    feats["url_length"] = len(url)
    feats["num_dots"] = url.count(".")
    feats["num_hyphens"] = url.count("-")
    feats["num_at"] = url.count("@")
    feats["num_qm"] = url.count("?")
    feats["num_slash"] = url.count("/")
    feats["num_digits"] = sum(ch.isdigit() for ch in url)
    feats["has_https"] = 1 if url.lower().startswith("https://") else 0
    feats["has_ip"] = has_ip_address(hostname)
    feats["hostname_length"] = len(hostname)
    feats["path_length"] = len(path)
    feats["num_subdomains"] = max(0, hostname.count(".") - 1)
    feats["suspicious_word_count"] = sum(1 for w in SUSPICIOUS_WORDS if w in full)
    feats["many_subdomains"] = 1 if feats["num_subdomains"] >= 3 else 0

    return feats
