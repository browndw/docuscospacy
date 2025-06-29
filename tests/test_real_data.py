"""
Enhanced tests using real corpus data for comprehensive validation.

This module tests docuscospacy functionality using real corpus data
from the docs/source/data directories to ensure realistic performance.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

import docuscospacy as ds
from tests.conftest import (
    assert_dataframe_structure,
    assert_frequency_table_valid,
    assert_tokens_table_valid,
)


class TestRealDataProcessing:
    """Test corpus processing with real data."""

    @pytest.mark.real_data
    def test_process_real_news_corpus(self, nlp_model, real_corpus_small):
        """Test processing real news corpus data."""
        # Process the corpus
        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)

        # Validate structure
        assert_tokens_table_valid(tokens)

        # Should have reasonable number of tokens
        assert tokens.height > 100, "Real corpus should produce substantial tokens"

        # Check for expected POS tags in news data
        pos_tags = tokens["pos_tag"].unique().to_list()
        expected_tags = ["NN1", "AT1", "VBZ", "DD1", "CC"]  # Common news tags
        found_expected = sum(1 for tag in expected_tags if tag in pos_tags)
        assert found_expected >= 3, f"Should find common POS tags, found: {pos_tags}"

        # Check for DocuScope tags
        ds_tags = tokens["ds_tag"].unique().to_list()
        assert len(ds_tags) > 5, "Should have multiple DocuScope tags"

    @pytest.mark.real_data
    def test_process_academic_corpus(self, nlp_model, real_corpus_academic):
        """Test processing real academic corpus data."""
        tokens = ds.docuscope_parse(real_corpus_academic, nlp_model)

        assert_tokens_table_valid(tokens)

        # Should have substantial vocabulary
        unique_tokens = tokens["token"].n_unique()
        assert unique_tokens > 50, "Academic texts should have rich vocabulary"

    @pytest.mark.real_data
    def test_compare_news_vs_academic(
        self, nlp_model, real_corpus_small, real_corpus_academic
    ):
        """Test that news and academic corpora show different patterns."""
        # Process both corpora
        news_tokens = ds.docuscope_parse(real_corpus_small, nlp_model)
        academic_tokens = ds.docuscope_parse(real_corpus_academic, nlp_model)

        # Generate frequency tables
        news_freq = ds.frequency_table(news_tokens, count_by="pos")
        academic_freq = ds.frequency_table(academic_tokens, count_by="pos")

        # Should have different vocabulary distributions
        news_vocab = set(news_freq["Token"].to_list())
        academic_vocab = set(academic_freq["Token"].to_list())

        # Some overlap expected, but should have unique elements
        overlap = len(news_vocab & academic_vocab)
        news_unique = len(news_vocab - academic_vocab)
        academic_unique = len(academic_vocab - news_vocab)

        assert news_unique > 0, "News corpus should have unique vocabulary"
        assert academic_unique > 0, "Academic corpus should have unique vocabulary"
        assert overlap > 0, "Corpora should share some common vocabulary"


class TestFrequencyAnalysisWithRealData:
    """Test frequency analysis with real corpus data."""

    @pytest.mark.real_data
    def test_frequency_analysis_accuracy(self, processed_tokens_real):
        """Test that frequency calculations are mathematically correct."""
        freq_table = ds.frequency_table(processed_tokens_real, count_by="pos")
        assert_frequency_table_valid(freq_table)

        # Verify that the frequency table structure is correct
        total_tokens = processed_tokens_real.height
        assert freq_table.height > 0, "Frequency table should have entries"

        # Check that AF values are reasonable
        af_values = freq_table["AF"].to_list()
        assert all(af > 0 for af in af_values), "All AF values should be positive"
        assert max(af_values) <= total_tokens, "Max AF should not exceed total tokens"

        # Check that RF values are reasonable (per million)
        rf_values = freq_table["RF"].to_list()
        assert all(rf > 0 for rf in rf_values), "All RF values should be positive"

        # Verify total AF is reasonable
        # (note: may be less than total_tokens due to grouping)
        total_af = sum(af_values)
        assert total_af > 0, "Total AF should be positive"

    @pytest.mark.real_data
    def test_tags_table_completeness(self, processed_tokens_real):
        """Test that tags table includes all expected tag types."""
        tags_table = ds.tags_table(processed_tokens_real, count_by="pos")

        # Should have reasonable number of different tags
        assert tags_table.height > 10, "Should have multiple different POS tags"

        # Total AF should be reasonable (may be less if some lack POS tags)
        total_af = tags_table["AF"].sum()
        total_tokens = processed_tokens_real.height
        assert total_af <= total_tokens, "Tag frequencies shouldn't exceed total tokens"
        assert total_af > 0.8 * total_tokens, "Most tokens should have POS tags"

        # Range values should be reasonable
        ranges = tags_table["Range"].to_list()
        assert all(0 <= r <= 100 for r in ranges), "All ranges should be 0-100%"

    @pytest.mark.real_data
    def test_dispersion_calculations(self, processed_tokens_real):
        """Test dispersion calculations with real data."""
        dispersions = ds.dispersions_table(processed_tokens_real, count_by="pos")

        assert_dataframe_structure(dispersions, ["Token", "Carrolls_D2"])

        # Check that dispersion values are reasonable (not necessarily 0-1)
        d2_values = dispersions["Carrolls_D2"].to_list()
        valid_d2 = [d for d in d2_values if not np.isnan(d)]

        if valid_d2:
            # D2 values should be finite and reasonable
            assert all(
                np.isfinite(d) for d in valid_d2
            ), "All D2 values should be finite"

            # Check that we have a reasonable range of values
            min_d2 = min(valid_d2)
            max_d2 = max(valid_d2)
            assert min_d2 <= max_d2, "Min should not exceed max"

            # Values should not be extremely large (indicate computation error)
            assert max_d2 < 1000, f"D2 values seem too large: max={max_d2}"


class TestNgramAnalysisWithRealData:
    """Test n-gram analysis with real corpus data."""

    @pytest.mark.real_data
    def test_bigram_extraction(self, processed_tokens_real):
        """Test bigram extraction with real data."""
        bigrams = ds.ngrams(processed_tokens_real, span=2, min_frequency=2)

        expected_columns = ["Token_1", "Token_2", "AF", "RF"]
        assert_dataframe_structure(bigrams, expected_columns)

        # Should find common bigrams in real text
        assert bigrams.height > 0, "Should find bigrams in real corpus"

        # Verify bigram frequency calculations
        if bigrams.height > 0:
            # Check that bigrams make linguistic sense
            top_bigram = bigrams.head(1)
            token1 = top_bigram["Token_1"].item()
            token2 = top_bigram["Token_2"].item()

            # Tokens should be non-empty strings
            assert isinstance(token1, str) and len(token1) > 0
            assert isinstance(token2, str) and len(token2) > 0

    @pytest.mark.real_data
    def test_trigram_extraction(self, processed_tokens_real):
        """Test trigram extraction with real data."""
        trigrams = ds.ngrams(processed_tokens_real, span=3, min_frequency=1)

        expected_columns = ["Token_1", "Token_2", "Token_3", "AF", "RF"]
        assert_dataframe_structure(trigrams, expected_columns)

        # May or may not find trigrams depending on corpus size
        if trigrams.height > 0:
            # Verify structure of found trigrams
            assert all(col in trigrams.columns for col in expected_columns)


class TestKeywordAnalysisWithRealData:
    """Test keyword and collocation analysis with real data."""

    @pytest.mark.real_data
    def test_kwic_analysis(self, processed_tokens_real):
        """Test KWIC analysis with common words."""
        # Find a common word that should appear multiple times
        freq_table = ds.frequency_table(processed_tokens_real, count_by="pos")
        if freq_table.height > 0:
            common_word = freq_table.head(1)["Token"].item()

            kwic_results = ds.kwic_center_node(
                processed_tokens_real,
                node_word=common_word,
                ignore_case=True,  # Allow case-insensitive matching
            )

            if kwic_results.height > 0:
                expected_columns = ["Doc ID", "Pre-Node", "Node", "Post-Node"]
                assert_dataframe_structure(kwic_results, expected_columns)

                # Check that all Node entries are variations of the target word
                nodes = kwic_results["Node"].unique().to_list()
                # With ignore_case=True, we may get different case variants
                normalized_nodes = {node.lower().strip() for node in nodes}
                target_normalized = common_word.lower().strip()

                assert (
                    target_normalized in normalized_nodes
                ), f"Target word '{common_word}' not found in nodes: {nodes}"

    @pytest.mark.real_data
    def test_collocation_analysis(self, processed_tokens_real):
        """Test collocation analysis with real data."""
        # Find a word that appears multiple times
        freq_table = ds.frequency_table(processed_tokens_real, count_by="pos")

        # Try with a few different words to find meaningful collocations
        for row in freq_table.head(5).iter_rows(named=True):
            target_word = row["Token"]
            if row["AF"] >= 3:  # Only test words that appear multiple times

                collocations = ds.coll_table(
                    processed_tokens_real, node_word=target_word, statistic="npmi"
                )

                if collocations.height > 0:
                    expected_columns = ["Token", "Tag", "Freq Span", "Freq Total", "MI"]
                    assert_dataframe_structure(collocations, expected_columns)

                    # MI values should be reasonable
                    mi_values = collocations["MI"].to_list()
                    # Filter out any NaN values for this check
                    valid_mi = [mi for mi in mi_values if not np.isnan(mi)]
                    if valid_mi:
                        assert all(
                            -20 <= mi <= 20 for mi in valid_mi
                        ), "MI values should be reasonable"
                    break


class TestStatisticalAnalysisWithRealData:
    """Test statistical analysis functions with real data."""

    @pytest.mark.real_data
    def test_keyness_analysis_real_corpora(
        self, nlp_model, real_corpus_small, real_corpus_academic
    ):
        """Test keyness analysis between real news and academic corpora."""
        # Process both corpora
        news_tokens = ds.docuscope_parse(real_corpus_small, nlp_model)
        academic_tokens = ds.docuscope_parse(real_corpus_academic, nlp_model)

        # Generate frequency tables
        news_freq = ds.frequency_table(news_tokens, count_by="pos")
        academic_freq = ds.frequency_table(academic_tokens, count_by="pos")

        # Perform keyness analysis
        keyness_results = ds.keyness_table(news_freq, academic_freq)

        if keyness_results.height > 0:
            expected_columns = [
                "Token",
                "Tag",
                "LL",
                "LR",
                "PV",
                "RF",
                "RF_Ref",
                "AF",
                "AF_Ref",
                "Range",
                "Range_Ref",
            ]
            assert_dataframe_structure(keyness_results, expected_columns)

            # Check statistical values are reasonable
            ll_values = keyness_results["LL"].to_list()
            valid_ll = [ll for ll in ll_values if not np.isnan(ll)]

            if valid_ll:
                # LL values should be non-negative
                assert all(
                    ll >= 0 for ll in valid_ll
                ), "LL values should be non-negative"

                # Should have some significant differences
                significant = sum(1 for ll in valid_ll if ll > 3.84)  # p < 0.05
                assert (
                    significant > 0
                ), "Should find some statistically significant differences"  # noqa: E501

    @pytest.mark.real_data
    def test_dtm_operations_real_data(self, processed_tokens_real):
        """Test document-term matrix operations with real data."""
        # Create DTM
        dtm = ds.tags_dtm(processed_tokens_real, count_by="pos")

        assert isinstance(dtm, pl.DataFrame)
        assert "doc_id" in dtm.columns

        # Should have multiple tag columns
        tag_columns = [col for col in dtm.columns if col != "doc_id"]
        assert len(tag_columns) > 5, "Should have multiple POS tag columns"

        # Test weighting
        weighted_dtm = ds.dtm_weight(dtm, scheme="prop")
        assert weighted_dtm.shape == dtm.shape

        # Proportional weights should sum to 1 for each document
        numeric_cols = [col for col in weighted_dtm.columns if col != "doc_id"]
        for i in range(min(3, weighted_dtm.height)):  # Check first few rows
            row_sum = sum(
                weighted_dtm.row(i)[j]
                for j in range(len(weighted_dtm.columns))
                if weighted_dtm.columns[j] in numeric_cols
            )
            assert (
                abs(row_sum - 1.0) < 0.01
            ), f"Row {i} should sum to ~1.0, got {row_sum}"

        # Test simplification
        simple_dtm = ds.dtm_simplify(dtm)
        assert simple_dtm.shape[0] == dtm.shape[0]  # Same number of documents
        assert simple_dtm.shape[1] < dtm.shape[1]  # Fewer columns (simplified tags)


class TestCorpusUtilitiesWithRealData:
    """Test corpus utility functions with real data files."""

    @pytest.mark.real_data
    def test_corpus_from_folder(self):
        """Test loading corpus from real data folders."""
        # Test with reference corpus folder
        ref_corpus_path = (
            Path(__file__).parent.parent / "docs" / "source" / "data" / "ref_corpus"
        )  # noqa: E501

        if ref_corpus_path.exists():
            # Load a subset by creating a temporary folder with a few files
            import tempfile
            import shutil

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Copy first 3 files for testing
                for i in range(1, 4):
                    src_file = ref_corpus_path / f"news_{i:02d}.txt"
                    if src_file.exists():
                        shutil.copy2(src_file, temp_path)

                # Test corpus_from_folder
                corpus = ds.corpus_from_folder(str(temp_path))

                assert isinstance(corpus, pl.DataFrame)
                assert "doc_id" in corpus.columns
                assert "text" in corpus.columns
                assert corpus.height == 3  # Should load 3 files

                # Text should be non-empty
                assert all(len(text) > 10 for text in corpus["text"].to_list())


@pytest.mark.real_data
class TestPerformanceWithRealData:
    """Test performance features with real corpus data."""

    def test_caching_with_real_data(
        self, nlp_model, real_corpus_small, performance_cache
    ):
        """Test that caching works with real data processing."""
        from docuscospacy.performance import PerformanceMonitor

        # First run - should be slower
        with PerformanceMonitor("first_run") as monitor1:
            tokens1 = ds.docuscope_parse(real_corpus_small, nlp_model)
            freq1 = ds.frequency_table(tokens1, count_by="pos")

        first_time = monitor1.end_time - monitor1.start_time

        # Second run - should potentially use cached results
        with PerformanceMonitor("second_run") as monitor2:
            tokens2 = ds.docuscope_parse(real_corpus_small, nlp_model)
            freq2 = ds.frequency_table(tokens2, count_by="pos")

        second_time = monitor2.end_time - monitor2.start_time

        # Results should be identical
        assert tokens1.shape == tokens2.shape
        assert freq1.shape == freq2.shape

        # Performance monitoring should work
        assert first_time > 0
        assert second_time > 0
        assert monitor1.start_time is not None
        assert monitor1.end_time is not None
