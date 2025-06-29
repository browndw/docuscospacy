"""
Configuration constants for docuscospacy.

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

from dataclasses import dataclass
import re


@dataclass
class ProcessingConfig:
    """Configuration constants for corpus processing."""

    # Text chunking
    MAX_CHUNK_SIZE: int = 500000
    CHUNK_ID_SEPARATOR: str = "@"

    # Normalization factors
    FREQUENCY_NORMALIZATION_FACTOR: int = 1000000

    # spaCy processing defaults
    DEFAULT_N_PROCESS: int = 1
    DEFAULT_BATCH_SIZE: int = 25

    # KWIC defaults
    KWIC_PRECEDING_CONTEXT: int = 7
    KWIC_FOLLOWING_CONTEXT: int = 7

    # Collocation defaults
    DEFAULT_PRECEDING_SPAN: int = 4
    DEFAULT_FOLLOWING_SPAN: int = 4

    # Validation
    EXPECTED_MODEL_PREFIX: str = "en_docusco_spacy"

    # Performance optimization settings
    ENABLE_CACHING: bool = True
    CACHE_MAX_SIZE: int = 128
    MEMORY_EFFICIENT_MODE: bool = False
    BATCH_PROCESSING_THRESHOLD: int = 1000  # rows
    PROGRESS_THRESHOLD: int = 5000  # rows to show progress

    # Memory optimization thresholds
    LARGE_CORPUS_THRESHOLD: int = 10000  # documents
    STREAMING_THRESHOLD: int = 50000  # documents


@dataclass
class RegexPatterns:
    """Compiled regex patterns for text processing."""

    SENTENCE_BOUNDARY = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")
    PUNCTUATION_ONLY = re.compile(r"^[!-/:-@\[-`{-~]+$")
    WORD_BOUNDARY = re.compile(r" ")
    CONTAINS_LETTERS = re.compile(r"[a-z]")
    POS_TAG_WITH_DIGITS = re.compile(r"\\d\\d$")
    POS_TAG_NON_FIRST_DIGIT = re.compile(r"[^1]$")
    CASE_INSENSITIVE_ITS = re.compile(r"(?i)^s$")
    CASE_INSENSITIVE_IT = re.compile(r"(?i)^it$")
    # For Polars string operations, we need the pattern as a string
    ITS_PATTERN_STR = r"(?i)\b(it)(s)\b"


# Global configuration instance
CONFIG = ProcessingConfig()
PATTERNS = RegexPatterns()
