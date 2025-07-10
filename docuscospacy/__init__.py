"""
docuscospacy: A comprehensive text analysis library for corpus linguistics.

This package provides tools for parsing, analyzing, and processing text corpora
using spaCy and advanced statistical methods. It includes modular analyzers,
performance optimizations, enhanced error handling, and comprehensive
validation.
"""

# Core analysis functions
from .corpus_analysis import (
    docuscope_parse,
    frequency_table,
    tags_table,
    dispersions_table,
    tags_dtm,
    ngrams,
    clusters_by_token,
    clusters_by_tag,
    kwic_center_node,
    coll_table,
    keyness_table,
    tag_ruler,
)

# Utility functions
from .corpus_utils import (
    get_text_paths,
    readtext,
    corpus_from_folder,
    dtm_weight,
    dtm_simplify,
    freq_simplify,
    tags_simplify,
    dtm_to_coo,
    from_tmtoolkit,
    convert_corpus,
)

# Modular analyzers and processors
from .analyzers import (
    FrequencyAnalyzer,
    TagAnalyzer,
    DispersionAnalyzer,
    NGramAnalyzer,
    ClusterAnalyzer,
    KWICAnalyzer,
    CollocationAnalyzer,
    KeynessAnalyzer,
    DocumentAnalyzer,
)

from .processors import (
    CorpusValidator,
    TextPreprocessor,
    TextChunker,
    SpacyProcessor,
    DataFrameTransformer,
    CorpusProcessor,
)

# Configuration
from .config import ProcessingConfig, RegexPatterns

# Performance utilities
from .performance import (
    PerformanceCache,
    MemoryOptimizer,
    ProgressTracker,
    PerformanceMonitor,
    cached_result,
    memory_efficient_join,
    optimize_polars_settings,
)

# Validation and error handling
from .validation import (
    # Exception classes
    DocuscoSpacyError,
    CorpusValidationError,
    ModelValidationError,
    DataFormatError,
    ParameterValidationError,
    FileSystemError,
    ValidationWarning,
    PerformanceWarning,
    # Validation functions
    validate_corpus_dataframe,
    validate_tokens_dataframe,
    validate_directory_path,
    validate_text_files_in_directory,
    validate_count_by_parameter,
    validate_span_parameter,
    validate_frequency_tables,
    warn_about_performance_settings,
    suggest_alternatives_for_empty_results,
)

# Package metadata
__version__ = "0.3.6"
__author__ = "David Brown"
__email__ = "dwb2@andrew.cmu.edu"

# Public API - define what gets imported with "from docuscospacy import *"
__all__ = [
    # Core analysis functions
    "docuscope_parse",
    "frequency_table",
    "tags_table",
    "dispersions_table",
    "tags_dtm",
    "ngrams",
    "clusters_by_token",
    "clusters_by_tag",
    "kwic_center_node",
    "coll_table",
    "keyness_table",
    "tag_ruler",
    # Utility functions
    "get_text_paths",
    "readtext",
    "corpus_from_folder",
    "dtm_weight",
    "dtm_simplify",
    "freq_simplify",
    "tags_simplify",
    "dtm_to_coo",
    "from_tmtoolkit",
    "convert_corpus",
    # Analyzers and processors
    "FrequencyAnalyzer",
    "TagAnalyzer",
    "DispersionAnalyzer",
    "NGramAnalyzer",
    "ClusterAnalyzer",
    "KWICAnalyzer",
    "CollocationAnalyzer",
    "KeynessAnalyzer",
    "DocumentAnalyzer",
    "CorpusValidator",
    "TextPreprocessor",
    "TextChunker",
    "SpacyProcessor",
    "DataFrameTransformer",
    "CorpusProcessor",
    # Configuration
    "ProcessingConfig",
    "RegexPatterns",
    # Performance utilities
    "PerformanceCache",
    "MemoryOptimizer",
    "ProgressTracker",
    "PerformanceMonitor",
    "cached_result",
    "memory_efficient_join",
    "optimize_polars_settings",
    # Validation and error handling
    "DocuscoSpacyError",
    "CorpusValidationError",
    "ModelValidationError",
    "DataFormatError",
    "ParameterValidationError",
    "FileSystemError",
    "ValidationWarning",
    "PerformanceWarning",
    "validate_corpus_dataframe",
    "validate_tokens_dataframe",
    "validate_directory_path",
    "validate_text_files_in_directory",
    "validate_count_by_parameter",
    "validate_span_parameter",
    "validate_frequency_tables",
    "warn_about_performance_settings",
    "suggest_alternatives_for_empty_results",
]
