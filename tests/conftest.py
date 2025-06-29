"""
Pytest configuration and shared fixtures for docuscospacy tests.

This module provides shared test fixtures, utilities, and configuration
for the docuscospacy test suite.
"""

import pytest
import polars as pl
import spacy
from pathlib import Path
import tempfile
import shutil

import docuscospacy as ds
from docuscospacy.performance import PerformanceCache


# Test data paths
DATA_DIR = Path(__file__).parent.parent / "docs" / "source" / "data"
REF_CORPUS_DIR = DATA_DIR / "ref_corpus"
TAR_CORPUS_DIR = DATA_DIR / "tar_corpus"


@pytest.fixture(scope="session")
def nlp_model():
    """Load spaCy model once per test session."""
    try:
        nlp = spacy.load("en_docusco_spacy")
        return nlp
    except Exception:
        pytest.skip("spaCy model 'en_docusco_spacy' not available")


@pytest.fixture(scope="session")
def simple_corpus():
    """Create a simple test corpus for basic functionality testing."""
    return pl.DataFrame(
        {
            "doc_id": ["test_doc1.txt", "test_doc2.txt", "test_doc3.txt"],
            "text": [
                "This is a test document for analysis.",
                "Another test document with different content.",
                "A third document to provide more test data.",
            ],
        }
    )


@pytest.fixture(scope="session")
def real_corpus_small():
    """Load a small subset of real corpus data for testing."""
    if not REF_CORPUS_DIR.exists():
        pytest.skip("Reference corpus data not available")

    # Load first 5 files from reference corpus
    corpus_data = []
    for i in range(1, 6):
        file_path = REF_CORPUS_DIR / f"news_{i:02d}.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            corpus_data.append({"doc_id": file_path.name, "text": content})

    if not corpus_data:
        pytest.skip("No reference corpus files found")

    return pl.DataFrame(corpus_data)


@pytest.fixture(scope="session")
def real_corpus_academic():
    """Load academic corpus data for comparative testing."""
    if not TAR_CORPUS_DIR.exists():
        pytest.skip("Academic corpus data not available")

    # Load first 5 files from academic corpus
    corpus_data = []
    for i in range(1, 6):
        file_path = TAR_CORPUS_DIR / f"acad_{i:02d}.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            corpus_data.append({"doc_id": file_path.name, "text": content})

    if not corpus_data:
        pytest.skip("No academic corpus files found")

    return pl.DataFrame(corpus_data)


@pytest.fixture(scope="session")
def processed_tokens_simple(nlp_model, simple_corpus):
    """Process simple corpus into tokens for reuse across tests."""
    return ds.docuscope_parse(simple_corpus, nlp_model)


@pytest.fixture(scope="session")
def processed_tokens_real(nlp_model, real_corpus_small):
    """Process real corpus into tokens for reuse across tests."""
    return ds.docuscope_parse(real_corpus_small, nlp_model)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def performance_cache(temp_cache_dir):
    """Create a performance cache instance for testing."""
    return PerformanceCache(cache_dir=temp_cache_dir)


@pytest.fixture
def corpus_with_issues():
    """Create a corpus with various edge cases for error testing."""
    return pl.DataFrame(
        {
            "doc_id": [
                "empty_doc.txt",
                "normal_doc.txt",
                "special_chars_doc.txt",
                "long_doc.txt",
                "unicode_doc.txt",
            ],
            "text": [
                "",  # Empty document
                "This is a normal document.",  # Normal document
                "Special chars: @#$%^&*()_+ {}[]|\\:;\"'<>,.?/~`",  # Special characters
                "Long document. " * 100,  # Very long document
                "Unicode test: café naïve résumé Москва 北京 العربية",  # Unicode
            ],
        }
    )


@pytest.fixture
def invalid_corpus_data():
    """Create various invalid corpus formats for error testing."""
    return {
        "missing_doc_id": pl.DataFrame({"text": ["Some text without doc_id"]}),
        "missing_text": pl.DataFrame({"doc_id": ["doc1.txt"]}),
        "wrong_types": pl.DataFrame(
            {
                "doc_id": [1, 2, 3],  # Should be strings
                "text": ["text1", "text2", "text3"],
            }
        ),
        "duplicate_ids": pl.DataFrame(
            {
                "doc_id": ["doc1.txt", "doc1.txt", "doc2.txt"],
                "text": ["text1", "text2", "text3"],
            }
        ),
        "null_values": pl.DataFrame(
            {"doc_id": ["doc1.txt", None, "doc3.txt"], "text": ["text1", "text2", None]}
        ),
    }


@pytest.fixture(scope="session")
def frequency_table_sample(processed_tokens_simple):
    """Generate a sample frequency table for testing."""
    return ds.frequency_table(processed_tokens_simple, count_by="pos")


@pytest.fixture(scope="session")
def tags_table_sample(processed_tokens_simple):
    """Generate a sample tags table for testing."""
    return ds.tags_table(processed_tokens_simple, count_by="pos")


@pytest.fixture
def large_corpus():
    """Create a larger corpus for performance testing."""
    corpus_data = []
    for i in range(50):
        corpus_data.append(
            {
                "doc_id": f"large_doc_{i:03d}.txt",
                "text": f"This is document number {i}. " + "Sample text content. " * 20,
            }
        )
    return pl.DataFrame(corpus_data)


# Test utilities
def assert_dataframe_structure(
    df: pl.DataFrame, expected_columns: list, min_rows: int = 0
):
    """Assert that a DataFrame has the expected structure."""
    assert isinstance(df, pl.DataFrame), "Result should be a polars DataFrame"
    assert df.height >= min_rows, f"DataFrame should have at least {min_rows} rows"

    for col in expected_columns:
        assert col in df.columns, f"Column '{col}' should be present"


def assert_frequency_table_valid(freq_table: pl.DataFrame):
    """Assert that a frequency table has valid structure and content."""
    required_columns = ["Token", "Tag", "AF", "RF", "Range"]
    assert_dataframe_structure(freq_table, required_columns)

    # Check data types and value ranges
    assert freq_table["AF"].dtype in [pl.UInt32, pl.Int64], "AF should be integer type"
    assert freq_table["RF"].dtype in [pl.Float64, pl.Float32], "RF should be float type"
    assert freq_table["Range"].dtype in [
        pl.Float64,
        pl.Float32,
    ], "Range should be float type"

    # Check value ranges
    if freq_table.height > 0:
        assert (freq_table["AF"] >= 0).all(), "All AF values should be non-negative"
        assert (freq_table["RF"] >= 0).all(), "All RF values should be non-negative"
        assert (
            freq_table["Range"] >= 0
        ).all(), "All Range values should be non-negative"
        assert (freq_table["Range"] <= 100).all(), "All Range values should be <= 100"


def assert_tokens_table_valid(tokens_table: pl.DataFrame):
    """Assert that a tokens table has valid structure and content."""
    required_columns = ["doc_id", "token", "pos_tag", "ds_tag"]
    assert_dataframe_structure(tokens_table, required_columns)

    # Check that we have actual tokens
    if tokens_table.height > 0:
        assert not tokens_table["token"].is_null().any(), "No tokens should be null"
        assert not tokens_table["doc_id"].is_null().any(), "No doc_ids should be null"


# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Fixture for performance monitoring in tests."""
    from docuscospacy.performance import PerformanceMonitor

    class TestPerformanceMonitor:
        def __init__(self):
            self.monitors = {}

        def time_operation(self, operation_name: str):
            return PerformanceMonitor(operation_name)

        def get_timing(self, operation_name: str):
            return self.monitors.get(operation_name, 0)

    return TestPerformanceMonitor()


# Hypothesis strategies for property-based testing
try:
    from hypothesis import strategies as st

    @st.composite
    def corpus_strategy(
        draw, min_docs=1, max_docs=10, min_text_length=10, max_text_length=200
    ):
        """Generate valid corpus DataFrames for property-based testing."""
        num_docs = draw(st.integers(min_value=min_docs, max_value=max_docs))

        doc_ids = []
        texts = []

        for i in range(num_docs):
            doc_id = f"doc_{i:03d}.txt"
            text_length = draw(
                st.integers(min_value=min_text_length, max_value=max_text_length)
            )
            text = draw(
                st.text(
                    min_size=text_length,
                    max_size=text_length,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")
                    ),
                )
            )

            doc_ids.append(doc_id)
            texts.append(text)

        return pl.DataFrame({"doc_id": doc_ids, "text": texts})

    # Make strategies available as fixtures
    @pytest.fixture
    def corpus_strategy_fixture():
        return corpus_strategy

except ImportError:
    # Hypothesis not available, provide dummy fixtures
    @pytest.fixture
    def corpus_strategy_fixture():
        pytest.skip("Hypothesis not available for property-based testing")


# Skip markers for conditional tests
def skip_if_no_model(nlp_model):
    """Skip test if spaCy model is not available."""
    if nlp_model is None:
        pytest.skip("spaCy model 'en_docusco_spacy' not available")


def skip_if_no_real_data():
    """Skip test if real corpus data is not available."""
    if not REF_CORPUS_DIR.exists() or not TAR_CORPUS_DIR.exists():
        pytest.skip("Real corpus data not available")
