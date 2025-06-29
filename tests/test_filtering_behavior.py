"""
Enhanced tests that account for docuscospacy's filtering behavior.

This module contains tests that properly handle the fact that certain
tags (like "Y" for punctuation) are filtered out during frequency analysis.
"""

import pytest
import polars as pl
import docuscospacy as ds
from pathlib import Path


class TestFilteringBehavior:
    """Test docuscospacy's filtering behavior for different tag types."""

    @pytest.fixture
    def simple_corpus(self):
        """Simple corpus for testing filtering behavior."""
        return pl.DataFrame(
            {
                "doc_id": ["test1.txt", "test2.txt"],
                "text": ["This is a test.", "Another test document!"],
            }
        )

    def test_token_vs_tag_count_with_filtering(self, simple_corpus, nlp_model):
        """Test that token and tag counts differ due to filtering."""
        # Parse the corpus
        tokens = ds.docuscope_parse(simple_corpus, nlp_model)

        # Count all tokens (including punctuation)
        total_tokens = tokens.height

        # Count tokens that would be included in frequency analysis
        # (excluding punctuation with pos_tag "Y")
        non_punctuation_tokens = tokens.filter(pl.col("pos_tag") != "Y").height

        # Generate frequency table
        freq_table = ds.frequency_table(tokens, count_by="pos")
        freq_count = freq_table.get_column("AF").sum()

        # Generate tag table
        tag_table = ds.tags_table(tokens, count_by="pos")
        tag_count = tag_table.get_column("AF").sum()

        # Both frequency and tag counts should match non-punctuation tokens
        assert freq_count == non_punctuation_tokens, (
            f"Frequency count ({freq_count}) should match non-punctuation tokens "
            f"({non_punctuation_tokens}), but total tokens is {total_tokens}"
        )

        assert tag_count == non_punctuation_tokens, (
            f"Tag count ({tag_count}) should match non-punctuation tokens "
            f"({non_punctuation_tokens}), but total tokens is {total_tokens}"
        )

        # Ensure we actually have some punctuation to test filtering
        punctuation_tokens = tokens.filter(pl.col("pos_tag") == "Y").height
        assert punctuation_tokens > 0, "Test should include punctuation tokens"

        # Verify the relationship
        assert total_tokens == non_punctuation_tokens + punctuation_tokens

    def test_punctuation_filtering_in_analysis(self, simple_corpus, nlp_model):
        """Test that punctuation is properly filtered from analysis functions."""
        tokens = ds.docuscope_parse(simple_corpus, nlp_model)

        # Check that punctuation tokens exist in raw output
        punctuation_tokens = tokens.filter(pl.col("pos_tag") == "Y")
        assert punctuation_tokens.height > 0, "Should have punctuation tokens"

        # Check that punctuation is filtered from frequency analysis
        freq_table = ds.frequency_table(tokens, count_by="pos")
        punct_in_freq = freq_table.filter(pl.col("Tag") == "Y")
        assert (
            punct_in_freq.height == 0
        ), "Punctuation should be filtered from frequency table"

        # Check other analysis functions also filter punctuation
        tag_table = ds.tags_table(tokens, count_by="pos")
        punct_in_tags = tag_table.filter(pl.col("Tag") == "Y")
        assert (
            punct_in_tags.height == 0
        ), "Punctuation should be filtered from tag table"

        # N-grams should also filter punctuation
        ngrams = ds.ngrams(tokens, span=2, min_frequency=0, count_by="pos")
        if ngrams.height > 0:  # Only check if we have ngrams
            # Should not have "Y" tags in the ngram results
            assert not any(
                "Y" in str(row) for row in ngrams.iter_rows()
            ), "N-grams should not include punctuation tokens"

    def test_filtered_tags_are_documented(self, simple_corpus, nlp_model):
        """Test that the filtering behavior is consistent and documented."""
        tokens = ds.docuscope_parse(simple_corpus, nlp_model)

        # Get all unique tags in the raw data
        all_tags = set(tokens.get_column("pos_tag").to_list())

        # Get tags that appear in frequency analysis
        freq_table = ds.frequency_table(tokens, count_by="pos")
        freq_tags = (
            set(freq_table.get_column("Tag").to_list())
            if freq_table.height > 0
            else set()
        )  # noqa: E501

        # Tags that are filtered out
        filtered_tags = all_tags - freq_tags

        # We expect "Y" (punctuation) to be filtered
        assert (
            "Y" in filtered_tags
        ), "Punctuation tag 'Y' should be filtered from analysis"

        # Log the filtering behavior for documentation
        print(f"Total tags in raw data: {len(all_tags)}")
        print(f"Tags in frequency analysis: {len(freq_tags)}")
        print(f"Filtered tags: {filtered_tags}")

        # Ensure filtering is consistent across functions
        tag_table = ds.tags_table(tokens, count_by="pos")
        tag_analysis_tags = (
            set(tag_table.get_column("Tag").to_list())
            if tag_table.height > 0
            else set()
        )  # noqa: E501

        assert (
            freq_tags == tag_analysis_tags
        ), "Frequency and tag analysis should filter the same tags"


class TestRealDataWithFiltering:
    """Test with real corpus data accounting for filtering."""

    @pytest.fixture
    def real_corpus_small(self):
        """Load a small sample of real corpus data."""
        docs_path = Path("docs/source/data/ref_corpus")
        if docs_path.exists():
            files = list(docs_path.glob("*.txt"))[:3]  # Just first 3 files
            if files:
                texts = []
                doc_ids = []
                for file in files:
                    try:
                        with open(file, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                            if text:  # Only include non-empty files
                                texts.append(text)
                                doc_ids.append(file.name)
                    except Exception:
                        continue

                if doc_ids:
                    return pl.DataFrame({"doc_id": doc_ids, "text": texts})

        # Fallback to synthetic data if real data not available
        return pl.DataFrame(
            {
                "doc_id": ["doc1.txt", "doc2.txt", "doc3.txt"],
                "text": [
                    "This is a longer test document with multiple sentences. It contains various punctuation marks!",  # noqa: E501
                    "Another document for testing purposes. This one has different content and structure.",  # noqa: E501
                    "The third document provides additional data for our analysis. It helps ensure robust testing.",  # noqa: E501
                ],
            }
        )

    def test_realistic_filtering_ratios(self, real_corpus_small, nlp_model):
        """Test filtering ratios with realistic corpus data."""
        if real_corpus_small.height == 0:
            pytest.skip("No corpus data available")

        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)

        # Count different token types
        total_tokens = tokens.height
        punctuation_tokens = tokens.filter(pl.col("pos_tag") == "Y").height
        analyzed_tokens = tokens.filter(pl.col("pos_tag") != "Y").height

        # Calculate ratios
        punct_ratio = punctuation_tokens / total_tokens if total_tokens > 0 else 0

        # Verify frequency analysis matches filtered count
        freq_table = ds.frequency_table(tokens, count_by="pos")
        freq_total = freq_table.get_column("AF").sum() if freq_table.height > 0 else 0

        assert freq_total == analyzed_tokens, (
            f"Frequency analysis total ({freq_total}) should match "
            f"non-punctuation tokens ({analyzed_tokens})"
        )

        # Log realistic ratios for documentation
        print(f"Total tokens: {total_tokens}")
        print(f"Punctuation tokens: {punctuation_tokens} ({punct_ratio:.1%})")
        print(f"Analyzed tokens: {analyzed_tokens} ({(1-punct_ratio):.1%})")

        # Reasonable expectations for real text
        assert (
            0.05 <= punct_ratio <= 0.25
        ), f"Punctuation ratio ({punct_ratio:.1%}) seems unrealistic"

    def test_consistency_across_analysis_functions(self, real_corpus_small, nlp_model):
        """Test that all analysis functions apply consistent filtering."""
        if real_corpus_small.height == 0:
            pytest.skip("No corpus data available")

        tokens = ds.docuscope_parse(real_corpus_small, nlp_model)
        non_punct_count = tokens.filter(pl.col("pos_tag") != "Y").height

        # Test various analysis functions
        freq_table = ds.frequency_table(tokens, count_by="pos")
        tag_table = ds.tags_table(tokens, count_by="pos")

        # All should have the same total count
        freq_total = freq_table.get_column("AF").sum() if freq_table.height > 0 else 0
        tag_total = tag_table.get_column("AF").sum() if tag_table.height > 0 else 0

        assert (
            freq_total == non_punct_count
        ), "Frequency table should match filtered count"
        assert tag_total == non_punct_count, "Tag table should match filtered count"

        # Test n-grams (should also filter consistently)
        ngrams = ds.ngrams(tokens, span=2, min_frequency=1, count_by="pos")
        if ngrams.height > 0:
            # N-grams count pairs, so total will be different, but should not include Y tags
            ngram_tags = []
            for col in ngrams.columns:
                if "Tag" in col:
                    ngram_tags.extend(ngrams.get_column(col).to_list())

            assert "Y" not in ngram_tags, "N-grams should not include punctuation tags"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
