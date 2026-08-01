from __future__ import annotations

"""Lightweight text feature helpers used before job-side embedding."""

import html
import re
from collections import Counter
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\s*[;:]\s+)|"
    r"(?:\s+--\s+)|"
    r"(?:\s+\u2013\s+)|"
    r"(?:\s+\u2014\s+)|"
    r"(?:\s*,\s+(?=(?:and|or|but|while|where|which|that|using|including|with|to)\b))",
    flags=re.IGNORECASE,
)


def tokenize_text(text: str) -> list[str]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _TOKEN_RE.findall(text.lower())


def build_stopword_set(
    custom_stopwords: Iterable[str] | None = None,
    *,
    use_stopword_filter: bool = True,
) -> set[str]:
    if not use_stopword_filter:
        return set()

    custom_sw = {word.lower() for word in (custom_stopwords or [])}
    return set(ENGLISH_STOP_WORDS) | custom_sw


def maybe_remove_stopwords(
    tokens: list[str],
    *,
    stopword_set: set[str],
    use_stopword_filter: bool,
) -> list[str]:
    if not use_stopword_filter:
        return tokens
    return [token for token in tokens if token not in stopword_set]


def maybe_remove_frequent_words(
    tokens: list[str],
    *,
    use_frequent_word_filter: bool,
    max_count: int | None,
) -> list[str]:
    if not use_frequent_word_filter or max_count is None:
        return tokens
    if max_count < 1:
        raise ValueError("max_count must be >= 1 when frequent-word filtering is enabled.")

    frequencies = Counter(tokens)
    return [token for token in tokens if frequencies[token] <= max_count]


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n([*-])\s*", r". \1 ", text)

    parts = _SENTENCE_SPLIT_RE.split(text)
    return [normalized for part in parts if (normalized := normalize_text(part))]


def split_clauses(sentence: str) -> list[str]:
    raw_parts = _CLAUSE_SPLIT_RE.split(sentence)
    parts = [normalized for part in raw_parts if (normalized := normalize_text(part))]

    merged: list[str] = []
    for part in parts:
        word_count = len(part.split())
        if merged and word_count < 4:
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)

    return merged


def phrase_chunks(
    text: str,
    min_words: int = 3,
    max_words: int = 24,
    include_sentences: bool = True,
    include_sentence_windows: bool = True,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentences = split_sentences(text)
    chunks: list[str] = []

    for sentence in sentences:
        clauses = split_clauses(sentence)

        for clause in clauses:
            word_count = len(clause.split())
            if min_words <= word_count <= max_words:
                chunks.append(clause)
            elif word_count > max_words:
                words = clause.split()
                for index in range(0, len(words), max_words):
                    subchunk = " ".join(words[index : index + max_words])
                    if min_words <= len(subchunk.split()) <= max_words:
                        chunks.append(subchunk)

        if include_sentences:
            word_count = len(sentence.split())
            if min_words <= word_count <= max_words:
                chunks.append(sentence)

    if include_sentence_windows and len(sentences) >= 2:
        for index in range(len(sentences) - 1):
            window = f"{sentences[index]} {sentences[index + 1]}".strip()
            word_count = len(window.split())

            if min_words <= word_count <= max_words:
                chunks.append(window)
            elif word_count > max_words:
                subchunk = " ".join(window.split()[:max_words])
                if len(subchunk.split()) >= min_words:
                    chunks.append(subchunk)

    seen = set()
    deduped: list[str] = []
    for chunk in chunks:
        key = chunk.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)

    return deduped
