from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.splitlines()]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line:
            current.append(line)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def html_to_text(value: str) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    for tag in soup.find_all(
        ["script", "style", "noscript", "svg", "form", "button", "iframe"]
    ):
        tag.decompose()
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all(
        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "figcaption"]
    ):
        tag.append("\n\n")
    return normalize_whitespace(soup.get_text("", strip=False))


def html_to_inline_text(value: str) -> str:
    if not value:
        return ""
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def count_words(value: str) -> int:
    """Count whitespace-delimited words, the project's canonical word metric."""
    return len(value.split())


def looks_english(text: str) -> bool:
    letters = [char for char in text[:10000] if char.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for char in letters if char.isascii())
    return ascii_letters / len(letters) >= 0.80


def content_hash(title: str, body_text: str) -> str:
    normalized = re.sub(r"\s+", " ", f"{title}\n{body_text}".lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
