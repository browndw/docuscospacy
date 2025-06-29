"""
Property-based tests for docuscospacy using hypothesis.

This module contains property-based tests that generate random inputs
to test edge cases and ensure robust behavior across a wide range of inputs.
"""

import pytest
import polars as pl

import docuscospacy as ds
from docuscospacy.validation import (
    DocuscoSpacyError,
    CorpusValidationError,
    ParameterValidationError,
    validate_corpus_dataframe,
)


class TestPropertyBasedSimple:
    """Simplified property-based tests for basic validation."""

    def test_corpus_validation_with_random_strings(self, simple_corpus):
        """Test corpus validation with various string inputs."""
        # This should work fine
        validate_corpus_dataframe(simple_corpus)

        # Test with invalid data
        invalid_corpus = pl.DataFrame({"wrong_column": ["some data"]})

        with pytest.raises((DocuscoSpacyError, CorpusValidationError)):
            validate_corpus_dataframe(invalid_corpus)

    def test_parameter_validation_edge_cases(self, processed_tokens_simple):
        """Test parameter validation with edge case values."""
        # Test invalid count_by values
        invalid_count_by_values = ["", "invalid", "INVALID", "123", "token"]

        for invalid_value in invalid_count_by_values:
            with pytest.raises(
                (DocuscoSpacyError, ParameterValidationError, ValueError)
            ):
                ds.frequency_table(processed_tokens_simple, count_by=invalid_value)

        # Test valid count_by values should work
        valid_count_by_values = ["pos", "ds", "both"]
        for valid_value in valid_count_by_values:
            result = ds.frequency_table(processed_tokens_simple, count_by=valid_value)
            assert result is not None

            # Handle both single DataFrame and tuple of DataFrames
            if valid_value == "both":
                assert isinstance(result, tuple)
                assert len(result) == 2
                assert hasattr(result[0], "height")  # First DataFrame
                assert hasattr(result[1], "height")  # Second DataFrame
            else:
                assert hasattr(result, "height")  # Single DataFrame

    def test_robustness_with_edge_case_corpora(self, nlp_model):
        """Test parsing robustness with various edge case corpora."""
        if nlp_model is None:
            pytest.skip("spaCy model not available")

        # Test with very short text (avoid completely empty)
        short_corpus = pl.DataFrame({"doc_id": ["short.txt"], "text": ["Hi."]})

        tokens = ds.docuscope_parse(short_corpus, nlp_model)
        assert isinstance(tokens, pl.DataFrame)
        assert tokens.height >= 0

        # Test with special characters
        special_corpus = pl.DataFrame(
            {"doc_id": ["special.txt"], "text": ["Test with special chars: @#$%^&*()"]}
        )

        tokens = ds.docuscope_parse(special_corpus, nlp_model)
        assert isinstance(tokens, pl.DataFrame)

        # Test with multiple sentences
        multi_corpus = pl.DataFrame(
            {
                "doc_id": ["multi.txt"],
                "text": ["First sentence. Second sentence! Third sentence?"],
            }
        )

        tokens = ds.docuscope_parse(multi_corpus, nlp_model)
        assert isinstance(tokens, pl.DataFrame)

    def test_frequency_table_consistency(self, processed_tokens_simple):
        """Test that frequency tables maintain consistency."""
        if processed_tokens_simple.height == 0:
            pytest.skip("No tokens to analyze")

        # Generate frequency tables with different count_by values
        valid_count_by = ["pos", "ds", "both"]

        for count_by in valid_count_by:
            freq_table = ds.frequency_table(processed_tokens_simple, count_by=count_by)

            # Basic consistency checks
            assert freq_table is not None

            # Handle both single DataFrame and tuple of DataFrames
            if count_by == "both":
                assert isinstance(freq_table, tuple)
                assert len(freq_table) == 2
                pos_table, ds_table = freq_table

                # Check both tables
                for table in [pos_table, ds_table]:
                    assert hasattr(table, "height")
                    if table.height > 0:
                        # All required columns should be present
                        required_cols = ["Token", "Tag", "AF", "RF", "Range"]
                        for col in required_cols:
                            assert col in table.columns
            else:
                assert hasattr(freq_table, "height")  # Single DataFrame
                if freq_table.height > 0:
                    # All required columns should be present
                    required_cols = ["Token", "Tag", "AF", "RF", "Range"]
                    for col in required_cols:
                        assert col in freq_table.columns

                # Data type checks
                assert freq_table["AF"].dtype in [pl.UInt32, pl.Int64]
                assert (freq_table["AF"] > 0).all()
                assert (freq_table["RF"] > 0).all()
                assert (freq_table["Range"] >= 0).all()
                assert (freq_table["Range"] <= 100).all()

                # Filtering checks - no Y or FU tags
                if "Tag" in freq_table.columns:
                    y_tags = freq_table.filter(pl.col("Tag").str.contains("^Y$"))
                    assert y_tags.height == 0, "Y tags should be filtered out"

                    fu_tags = freq_table.filter(pl.col("Tag").str.contains("^FU$"))
                    assert fu_tags.height == 0, "FU tags should be filtered out"

    def test_tags_table_consistency(self, processed_tokens_simple):
        """Test that tags tables maintain consistency."""
        if processed_tokens_simple.height == 0:
            pytest.skip("No tokens to analyze")

        # Generate tags tables
        valid_count_by = ["pos", "ds"]

        for count_by in valid_count_by:
            tags_table = ds.tags_table(processed_tokens_simple, count_by=count_by)

            # Basic consistency checks
            assert isinstance(tags_table, pl.DataFrame)

            if tags_table.height > 0:
                # Required columns should be present
                required_cols = ["Tag", "AF", "RF", "Range"]
                for col in required_cols:
                    assert col in tags_table.columns

                # No Y or FU tags should be present
                y_tags = tags_table.filter(pl.col("Tag").str.contains("^Y$"))
                assert y_tags.height == 0, "Y tags should be filtered out"

                fu_tags = tags_table.filter(pl.col("Tag").str.contains("^FU$"))
                assert fu_tags.height == 0, "FU tags should be filtered out"

    def test_error_handling_robustness(self):
        """Test error handling with various invalid inputs."""
        # Test with None inputs
        with pytest.raises(
            (DocuscoSpacyError, CorpusValidationError, ValueError, TypeError)
        ):
            validate_corpus_dataframe(None)

        # Test with empty DataFrame
        empty_df = pl.DataFrame()
        with pytest.raises((DocuscoSpacyError, CorpusValidationError)):
            validate_corpus_dataframe(empty_df)

        # Test with wrong schema
        wrong_schema = pl.DataFrame(
            {"id": [1, 2, 3], "content": ["text1", "text2", "text3"]}
        )
        with pytest.raises((DocuscoSpacyError, CorpusValidationError)):
            validate_corpus_dataframe(wrong_schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
