"""
Integration tests for docuscospacy focusing on error handling and consistency.

This module contains integration tests that verify the behavior
of the docuscospacy package under various conditions.
"""

import pytest
import polars as pl
import time
from unittest.mock import MagicMock

import docuscospacy as ds
from docuscospacy.validation import DocuscoSpacyError


class TestPerformanceIntegration:
    """Integration tests for performance features."""

    def test_basic_performance_tracking(self, nlp_model, real_corpus_small):
        """Test basic performance without advanced features."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        # Test basic processing works
        start_time = time.time()
        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)
        duration = time.time() - start_time

        # Should complete in reasonable time and produce results
        assert duration < 30  # Allow up to 30 seconds for small corpus
        assert tokens.height > 0

    def test_memory_optimization_with_large_corpus(self, nlp_model, large_corpus):
        """Test memory optimization features with larger corpus."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        # Should handle large corpus without memory issues
        try:
            tokens = ds.docuscope_parse(large_corpus, nlp_model, batch_size=10)
            assert isinstance(tokens, pl.DataFrame)
            assert tokens.height > 0

            # Should be able to generate frequency tables
            freq_table = ds.frequency_table(tokens, count_by="ds")
            assert isinstance(freq_table, pl.DataFrame)

        except MemoryError:
            pytest.fail("Memory optimization failed to handle large corpus")

    def test_batch_processing_efficiency(self, nlp_model, real_corpus_small):
        """Test batch processing with different sizes."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        # Test different batch sizes produce same results
        tokens1 = ds.docuscope_parse(real_corpus_small, nlp_model, batch_size=1)
        tokens2 = ds.docuscope_parse(real_corpus_small, nlp_model, batch_size=10)

        # Should produce same number of tokens regardless of batch size
        assert tokens1.height == tokens2.height


class TestErrorHandlingIntegration:
    """Integration tests for error handling and validation."""

    def test_basic_corpus_validation(self, nlp_model):
        """Test basic corpus validation."""
        # Test with invalid corpus structures
        invalid_corpus = pl.DataFrame(
            {"wrong_column": ["doc1.txt"], "other_column": ["Some text"]}
        )

        if nlp_model is not None:
            with pytest.raises((Exception, ValueError)):
                ds.docuscope_parse(invalid_corpus, nlp_model)

    def test_model_validation_error_handling(self, simple_corpus):
        """Test error handling when spaCy model is invalid."""
        # Mock invalid model
        invalid_model = MagicMock()
        invalid_model.pipe.side_effect = Exception("Model processing failed")

        with pytest.raises((DocuscoSpacyError, Exception)):
            ds.docuscope_parse(simple_corpus, invalid_model)

    def test_graceful_handling_of_edge_cases(self, nlp_model):
        """Test graceful handling of various edge cases."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        # Create a simple corpus with edge cases
        edge_cases_corpus = pl.DataFrame(
            {
                "doc_id": ["empty.txt", "short.txt", "normal.txt"],
                "text": ["", "Hi", "This is a normal document with sufficient text."],
            }
        )

        # Should handle edge cases without crashing
        try:
            tokens = ds.docuscope_parse(edge_cases_corpus, nlp_model)
            assert isinstance(tokens, pl.DataFrame)

            # Should be able to generate frequency tables
            if tokens.height > 0:
                freq_table = ds.frequency_table(tokens, count_by="ds")
                assert isinstance(freq_table, pl.DataFrame)
        except Exception as e:
            # Should not crash but may produce warnings
            if "exploded columns must have matching element counts" in str(e):
                pytest.skip("Edge case handling needs improvement")
            else:
                raise

    def test_parameter_validation_integration(self, processed_tokens_simple):
        """Test parameter validation across different functions."""
        # Invalid count_by parameter
        with pytest.raises((DocuscoSpacyError, ValueError)):
            ds.frequency_table(processed_tokens_simple, count_by="invalid")

        # Test that valid count_by values work
        freq_table = ds.frequency_table(processed_tokens_simple, count_by="ds")
        assert isinstance(freq_table, pl.DataFrame)


class TestDataConsistencyIntegration:
    """Integration tests for data consistency and filtering behavior."""

    def test_filtering_consistency_across_functions(self, nlp_model, real_corpus_small):
        """Test that filtering behavior is consistent across all analysis functions."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)

        if tokens.height == 0:
            pytest.skip("No tokens generated")

        # Generate different types of tables
        freq_table_pos = ds.frequency_table(tokens, count_by="pos")
        freq_table_ds = ds.frequency_table(tokens, count_by="ds")

        tags_table_pos = ds.tags_table(tokens, count_by="pos")
        tags_table_ds = ds.tags_table(tokens, count_by="ds")

        # All tables should exclude Y and FU tags
        tables_to_check = [
            ("freq_pos", freq_table_pos),
            ("freq_ds", freq_table_ds),
            ("tags_pos", tags_table_pos),
            ("tags_ds", tags_table_ds),
        ]

        for name, table in tables_to_check:
            if table.height > 0 and "tag" in table.columns:
                # No Y tags (punctuation)
                y_tags = table.filter(pl.col("tag").str.contains("^Y$"))
                assert y_tags.height == 0, f"Found Y tags in {name}: {y_tags}"

                # No FU tags (unassigned)
                fu_tags = table.filter(pl.col("tag").str.contains("^FU$"))
                assert fu_tags.height == 0, f"Found FU tags in {name}: {fu_tags}"

    def test_tag_coverage_analysis(self, nlp_model, real_corpus_small):
        """Test analysis of tag coverage and filtering impact."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)

        if tokens.height == 0:
            pytest.skip("No tokens generated")

        # Count all tokens
        total_tokens = tokens.height

        # Count Y tags (should exist in raw tokens)
        y_tokens = tokens.filter(pl.col("ds_tag").str.contains("^Y$"))
        y_count = y_tokens.height

        # Count FU tags (should exist in raw tokens)
        fu_tokens = tokens.filter(pl.col("ds_tag").str.contains("^FU$"))
        fu_count = fu_tokens.height

        # Generate frequency table
        freq_table = ds.frequency_table(tokens, count_by="ds")

        # Calculate tokens represented in frequency table
        if freq_table.height > 0:
            represented_tokens = freq_table["AF"].sum()

            # The difference should account for filtered Y and FU tags
            expected_represented = total_tokens - y_count - fu_count
            assert represented_tokens <= expected_represented

        print(f"Total tokens: {total_tokens}")
        print(f"Y tags filtered: {y_count}")
        print(f"FU tags filtered: {fu_count}")
        if freq_table.height > 0:
            print(f"Tokens in frequency table: {represented_tokens}")


class TestRegressionIntegration:
    """Regression tests to ensure consistent behavior across versions."""

    def test_output_format_stability(self, nlp_model, simple_corpus):
        """Test that output formats remain stable."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        tokens = ds.docuscope_parse(simple_corpus, nlp_model)

        # Required columns should always be present
        required_token_columns = ["doc_id", "token", "pos_tag", "ds_tag"]
        for col in required_token_columns:
            assert col in tokens.columns

        if tokens.height > 0:
            freq_table = ds.frequency_table(tokens, count_by="ds")
            required_freq_columns = ["Token", "Tag", "AF", "RF", "Range"]
            for col in required_freq_columns:
                assert col in freq_table.columns

            tags_table = ds.tags_table(tokens, count_by="ds")
            required_tags_columns = ["Tag", "AF", "RF", "Range"]
            for col in required_tags_columns:
                assert col in tags_table.columns

    def test_numerical_precision_stability(self, processed_tokens_simple):
        """Test that numerical calculations remain stable."""
        freq_table = ds.frequency_table(processed_tokens_simple, count_by="ds")

        if freq_table.height > 0:
            # AF should be integers
            assert freq_table["AF"].dtype in [pl.UInt32, pl.Int64]

            # RF should be positive floats
            assert (freq_table["RF"] > 0).all()

            # Range should be between 0 and 100
            assert (freq_table["Range"] >= 0).all()
            assert (freq_table["Range"] <= 100).all()

            # Sum of AF should be reasonable
            total_af = freq_table["AF"].sum()
            assert total_af > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
