"""
Analyzer classes for corpus linguistic analysis.

This module provides specialized analyzer classes that encapsulate
corpus analysis functionality for better organization, reusability,
and maintainability. Each analyzer focuses on a specific type of
linguistic analysis with comprehensive caching and validation.

Classes:
    FrequencyAnalyzer: Frequency analysis and distribution statistics
    TagAnalyzer: Part-of-speech and DocuScope tag analysis
    DispersionAnalyzer: Token dispersion and distribution analysis
    NGramAnalyzer: N-gram extraction and analysis
    ClusterAnalyzer: Token and tag clustering analysis
    KWICAnalyzer: Keywords-in-context concordance analysis
    CollocationAnalyzer: Collocation and association analysis
    KeynessAnalyzer: Keyness and distinctiveness analysis
    DocumentAnalyzer: Document-level analysis and transformations

Example:
    Basic frequency analysis::

        import polars as pl
        from docuscospacy.analyzers import FrequencyAnalyzer

        # Assume you have processed tokens
        tokens = pl.DataFrame({
            'doc_id': ['doc1'] * 3,
            'token': ['hello', 'world', 'hello'],
            'pos_tag': ['UH', 'NN', 'UH'],
            'ds_tag': ['B-Greeting', 'B-Object', 'B-Greeting']
        })

        analyzer = FrequencyAnalyzer()
        freq_table = analyzer.frequency_table(tokens)

    Using multiple analyzers together::

        from docuscospacy.analyzers import (
            FrequencyAnalyzer, TagAnalyzer, NGramAnalyzer
        )

        freq_analyzer = FrequencyAnalyzer()
        tag_analyzer = TagAnalyzer()
        ngram_analyzer = NGramAnalyzer()

        # Get frequency distributions
        frequencies = freq_analyzer.frequency_table(tokens)

        # Analyze tag patterns
        tag_patterns = tag_analyzer.tags_table(tokens)

        # Extract bigrams
        bigrams = ngram_analyzer.ngrams(tokens, n=2)

.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

from collections import OrderedDict
from typing import Union, Tuple
import numpy as np
import polars as pl
from scipy.stats.distributions import chi2

from .performance import cached_result, MemoryOptimizer, PerformanceMonitor
from .validation import validate_tokens_dataframe, validate_count_by_parameter


class FrequencyAnalyzer:
    """
    Handles frequency analysis and distribution statistics.

    This analyzer provides methods for computing frequency tables, relative
    frequencies, ranges, and related statistics for tokens in a corpus.
    All methods support caching for improved performance on repeated analyses.

    Attributes:
        None (all methods are instance methods for consistency)

    Example:
        Basic frequency analysis::

            import polars as pl
            from docuscospacy.analyzers import FrequencyAnalyzer

            # Sample tokens data
            tokens = pl.DataFrame({
                'doc_id': ['doc1', 'doc1', 'doc2', 'doc2'],
                'token': ['hello', 'world', 'hello', 'test'],
                'pos_tag': ['UH', 'NN', 'UH', 'NN'],
                'ds_tag': ['B-Greeting', 'B-Object', 'B-Greeting', 'B-Test']
            })

            analyzer = FrequencyAnalyzer()

            # Get frequency table
            freq_table = analyzer.frequency_table(tokens)
            print(freq_table)
            # Output includes Token, Tag, AF (absolute freq), RF (relative freq), Range

        Filter by frequency::

            # Only show tokens with AF >= 2
            frequent_tokens = analyzer.frequency_table(tokens, af_min=2)

        Count by tags instead of tokens::

            # Analyze tag frequencies
            tag_freq = analyzer.frequency_table(tokens, count_by='Tag')
    """

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validate_tokens_dataframe(tokens_table, "in FrequencyAnalyzer")

    @staticmethod
    def _validate_count_by(count_by: str, valid_types: list) -> None:
        """Validate count_by parameter."""
        validate_count_by_parameter(count_by, valid_types, "in FrequencyAnalyzer")

    @staticmethod
    def _summarize_counts(df: pl.DataFrame) -> pl.DataFrame:
        """Common summary statistics for frequency tables."""
        return (
            df.pivot(index="Token", on="doc_id", values="len", aggregate_function="sum")
            .with_columns(pl.all().exclude("Token").cast(pl.UInt32, strict=True))
            # calculate range
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().is_not_null()).alias("Range")
            )
            .with_columns(pl.selectors.numeric().fill_null(strategy="zero"))
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range")
                .truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude("Range").is_not_null()
                    )
                )
                .mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().exclude("Range")).alias("AF")
            )
            .sort("AF", descending=True)
            .select(["Token", "AF", "Range"])
            # calculate relative frequency
            .with_columns(pl.col("AF").truediv(pl.sum("AF")).mul(1000000).alias("RF"))
            # format data
            .unnest("Token")
            .select(["Token", "Tag", "AF", "RF", "Range"])
        )

    @cached_result
    def frequency_table(
        self,
        tokens_table: pl.DataFrame,
        count_by: str = "pos",
        af_min: int = 1,
        af_max: int = None,
        rf_min: float = 0.0,
        rf_max: float = None,
        range_min: float = 0.0,
        range_max: float = 100.0,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate a comprehensive frequency table for tokens or tags.

        Creates frequency distributions with absolute frequencies (AF), relative
        frequencies (RF) per million tokens, and range (percentage of documents
        containing each token). Supports filtering by frequency thresholds.

        Args:
            tokens_table: A polars DataFrame as generated by docuscope_parse,
                containing 'doc_id', 'token', 'pos_tag', and 'ds_tag' columns.
            count_by: Aggregation method - 'pos' for POS tags, 'ds' for DocuScope
                tags, or 'both' to return separate tables for each.
            af_min: Minimum absolute frequency threshold (inclusive).
            af_max: Maximum absolute frequency threshold (inclusive). If None,
                no upper limit is applied.
            rf_min: Minimum relative frequency threshold (per million tokens).
            rf_max: Maximum relative frequency threshold (per million tokens).
                If None, no upper limit is applied.
            range_min: Minimum range threshold (percentage of documents).
            range_max: Maximum range threshold (percentage of documents).

        Returns:
            If count_by is 'pos' or 'ds': A polars DataFrame with columns:
                - Token: The token text
                - Tag: The part-of-speech or DocuScope tag
                - AF: Absolute frequency (raw count)
                - RF: Relative frequency (per million tokens)
                - Range: Percentage of documents containing this token

            If count_by is 'both': A tuple of (pos_table, ds_table) where each
                table has the same structure as above.

        Raises:
            ValidationError: If tokens_table doesn't have required columns or
                if count_by is not one of the valid options.

        Example:
            Basic frequency analysis::

                analyzer = FrequencyAnalyzer()
                freq_table = analyzer.frequency_table(tokens)
                print(freq_table.head())
                # Token    Tag    AF    RF      Range
                # the      DT     245   12500.3 89.2
                # and      CC     198   10100.1 78.5

            Filter by frequency::

                # Only tokens appearing at least 5 times
                frequent = analyzer.frequency_table(tokens, af_min=5)

                # Only tokens appearing in at least 25% of documents
                widespread = analyzer.frequency_table(tokens, range_min=25.0)

            Analyze both POS and DocuScope tags::

                pos_freq, ds_freq = analyzer.frequency_table(tokens, count_by='both')

        Note:
            - Results are automatically cached for repeated calls with same parameters
            - Large corpora are automatically optimized for memory efficiency
            - RF is calculated as: (token_count / total_tokens) * 1,000,000
            - Range is calculated as: (docs_with_token / total_docs) * 100
        """
        with PerformanceMonitor(f"Frequency analysis ({count_by})"):
            count_types = ["pos", "ds", "both"]
            self._validate_count_by(count_by, count_types)
            self._validate_tokens_table(tokens_table)

            # Check if we need memory optimization
            if MemoryOptimizer.is_large_corpus(tokens_table):
                return self._frequency_table_memory_optimized(tokens_table, count_by)

            # Standard processing
            return self._frequency_table_standard(tokens_table, count_by)

    def _frequency_table_standard(
        self, tokens_table: pl.DataFrame, count_by: str
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Standard frequency table processing."""
        # format tokens and sum by doc_id
        df_pos = (
            tokens_table.group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
            .agg(pl.col("token").str.join(""))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .filter(pl.col("pos_tag") != "Y")
            .rename({"pos_tag": "Tag"})
            .rename({"token": "Token"})
            .group_by(["doc_id", "Token", "Tag"])
            .len()
            .with_columns(pl.struct(["Token", "Tag"]))
            .select(pl.exclude("Tag"))
        )

        df_pos = self._summarize_counts(df_pos).sort(
            ["AF", "Token"], descending=[True, False]
        )

        if count_by == "pos":
            return df_pos
        else:
            df_ds = (
                tokens_table.filter(
                    ~(
                        pl.col("token").str.contains("^[[[:punct:]] ]+$")
                        & pl.col("ds_tag").str.contains("Untagged")
                    )
                )
                .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
                .agg(pl.col("token").str.join(""))
                .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
                .rename({"ds_tag": "Tag"})
                .rename({"token": "Token"})
                .group_by(["doc_id", "Token", "Tag"])
                .len()
                .with_columns(pl.struct(["Token", "Tag"]))
                .select(pl.exclude("Tag"))
            )

            df_ds = self._summarize_counts(df_ds).sort(
                ["AF", "Token"], descending=[True, False]
            )

        if count_by == "ds":
            return df_ds
        else:
            return df_pos, df_ds

    def _frequency_table_memory_optimized(
        self, tokens_table: pl.DataFrame, count_by: str
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Memory-optimized frequency table processing for large corpora."""
        return MemoryOptimizer.batch_process(
            tokens_table, self._frequency_table_standard, count_by=count_by
        )


class TagAnalyzer:
    """Handles tag frequency analysis and document-term matrices."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    @staticmethod
    def _validate_count_by(count_by: str, valid_types: list) -> None:
        """Validate count_by parameter."""
        if count_by not in valid_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {valid_types}")

    @staticmethod
    def _summarize_tag_counts(df: pl.DataFrame) -> pl.DataFrame:
        """Common summary statistics for tag tables."""
        return (
            df.pivot(index="Tag", on="doc_id", values="len", aggregate_function="sum")
            .with_columns(pl.all().exclude("Tag").cast(pl.UInt32, strict=True))
            # calculate range
            .with_columns(Range=pl.sum_horizontal(pl.selectors.numeric().is_not_null()))
            .with_columns(pl.selectors.numeric().fill_null(strategy="zero"))
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range")
                .truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude("Range").is_not_null()
                    )
                )
                .mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().exclude("Range")).alias("AF")
            )
            .sort("AF", descending=True)
            .select(["Tag", "AF", "Range"])
            # calculate relative frequency
            .with_columns(pl.col("AF").truediv(pl.sum("AF")).mul(100).alias("RF"))
            .select(["Tag", "AF", "RF", "Range"])
        )

    @cached_result
    def tags_table(
        self, tokens_table: pl.DataFrame, count_by: str = "pos"
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate a count of tag frequencies.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
        :return: A polars DataFrame of absolute frequencies,
            normalized frequencies and ranges
        """
        count_types = ["pos", "ds", "both"]
        self._validate_count_by(count_by, count_types)
        self._validate_tokens_table(tokens_table)

        # format tokens and sum by doc_id
        df_pos = (
            tokens_table.filter(pl.col("pos_tag") != "Y")
            .group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
            .first()
            .group_by(["doc_id", "pos_tag"])
            .len()
            .rename({"pos_tag": "Tag"})
        )

        df_pos = self._summarize_tag_counts(df_pos).sort(
            ["AF", "Tag"], descending=[True, False]
        )

        if count_by == "pos":
            return df_pos
        else:
            df_ds = (
                tokens_table.filter(
                    ~(
                        pl.col("token").str.contains("^[[[:punct:]] ]+$")
                        & pl.col("ds_tag").str.contains("Untagged")
                    )
                )
                .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
                .first()
                .group_by(["doc_id", "ds_tag"])
                .len()
                .rename({"ds_tag": "Tag"})
            )

            df_ds = self._summarize_tag_counts(df_ds).sort(
                ["AF", "Tag"], descending=[True, False]
            )

        if count_by == "ds":
            return df_ds
        else:
            return df_pos, df_ds

    @cached_result
    def tags_dtm(
        self, tokens_table: pl.DataFrame, count_by: str = "pos"
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """
        Generate a document-term matrix of raw tag counts.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
        :return: A polars DataFrame of absolute tag frequencies
            for each document
        """
        count_types = ["pos", "ds", "both"]
        self._validate_count_by(count_by, count_types)
        self._validate_tokens_table(tokens_table)

        df_pos = (
            tokens_table.filter(pl.col("pos_tag") != "Y")
            .group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
            .first()
            .group_by(["doc_id", "pos_tag"])
            .len()
            .rename({"pos_tag": "tag"})
            .with_columns(pl.col("len").sum().over("tag").alias("total"))
            .sort(["total", "doc_id"], descending=[True, False])
            .pivot(index="doc_id", on="tag", values="len", aggregate_function="sum")
            .fill_null(strategy="zero")
        )

        if count_by == "pos":
            return df_pos

        df_ds = (
            tokens_table.filter(
                ~(
                    pl.col("token").str.contains("^[[[:punct:]] ]+$")
                    & pl.col("ds_tag").str.contains("Untagged")
                )
            )
            .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
            .first()
            .group_by(["doc_id", "ds_tag"])
            .len()
            .rename({"ds_tag": "tag"})
            .with_columns(pl.col("len").sum().over("tag").alias("total"))
            .sort(["total", "doc_id"], descending=[True, False])
            .pivot(index="doc_id", on="tag", values="len", aggregate_function="sum")
            .fill_null(strategy="zero")
        )

        if count_by == "ds":
            return df_ds
        else:
            return df_pos, df_ds


class DispersionAnalyzer:
    """Handles dispersion analysis and related calculations."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    @staticmethod
    def _calc_disp2(v: pl.Series, total: float, s=None) -> pl.DataFrame:
        """
        Calculate dispersion measures for a polars Series.

        :param v: A polars Series or column of a DataFrame
        :param total: The total number of tokens in a corpus
        :param s: Optional size parameter
        :return: A polars DataFrame with dispersion measures
        """
        token_tag = v.name
        token_tag = token_tag.rsplit("_", 1)

        v = v.to_numpy()
        if s is None:
            s = np.ones(len(v)) / len(v)
        n = len(v)  # n
        f = v.sum()  # f
        s = s / np.sum(s)  # s

        nf = 1000000

        values = {}
        values["Token"] = token_tag[0]
        values["Tag"] = token_tag[1]
        values["AF"] = f
        # normalizing according to the normalizing factor 'nf'
        values["RF"] = (f / total) * nf
        values["Carrolls_D2"] = (
            np.log2(f) - (np.sum(v[v != 0] * np.log2(v[v != 0])) / f)
        ) / np.log2(n)
        values["Rosengrens_S"] = np.sum(np.sqrt(v * s)) ** 2 / f
        values["Lynes_D3"] = 1 - (
            np.sum(((v - np.mean(v)) ** 2) / np.mean(v)) / (4 * f)
        )
        values["DC"] = ((np.sum(np.sqrt(v)) / n) ** 2) / np.mean(v)
        values["Juillands_D"] = 1 - (np.std(v / s) / np.mean(v / s)) / np.sqrt(
            len(v / s) - 1
        )

        values["DP"] = np.sum(np.abs((v / f) - s)) / 2
        # corrected
        values["DP_norm"] = (np.sum(np.abs((v / f) - s)) / 2) / (1 - np.min(s))

        return pl.DataFrame(values)

    def dispersions_table(
        self, tokens_table: pl.DataFrame, count_by: str = "pos"
    ) -> pl.DataFrame:
        """
        Generate a table of dispersion measures.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :return: A polars DataFrame with various dispersion measures
        """
        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)

        if count_by == "pos":
            dtm = (
                tokens_table.with_columns(
                    pl.col("token").str.to_lowercase().str.strip_chars()
                )
                .filter(pl.col("pos_tag") != "Y")
                .group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
                .first()
                .with_columns(
                    pl.concat_str(
                        [pl.col("token"), pl.col("pos_tag")], separator="_"
                    ).alias("pos_tag")
                )
                .group_by(["doc_id", "pos_tag"])
                .len()
                .with_columns(pl.col("len").sum().over("pos_tag").alias("total"))
                .sort(["total", "doc_id"], descending=[True, False])
                .pivot(
                    index="doc_id", on="pos_tag", values="len", aggregate_function="sum"
                )
                .fill_null(strategy="zero")
            )

        if count_by == "ds":
            dtm = (
                tokens_table.with_columns(
                    pl.col("token").str.to_lowercase().str.strip_chars()
                )
                .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
                .first()
                .with_columns(
                    pl.concat_str(
                        [pl.col("token"), pl.col("ds_tag")], separator="_"
                    ).alias("ds_tag")
                )
                .group_by(["doc_id", "ds_tag"])
                .len()
                .with_columns(pl.col("len").sum().over("ds_tag").alias("total"))
                .sort(["total", "doc_id"], descending=[True, False])
                .pivot(
                    index="doc_id", on="ds_tag", values="len", aggregate_function="sum"
                )
                .fill_null(strategy="zero")
            )

        total = dtm.drop("doc_id").sum().sum_horizontal().item()
        parts = (dtm.drop("doc_id").sum_horizontal() / total).to_numpy()
        idx = range(1, dtm.width)
        dsp = [self._calc_disp2(dtm[:, i], total, parts) for i in idx]
        dsp = pl.concat(dsp)
        return dsp


class NGramAnalyzer:
    """Handles n-gram analysis and clustering operations."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    @staticmethod
    def _validate_span(span: int) -> None:
        """Validate span parameter."""
        if span < 2 or span > 5:
            raise ValueError("Span must be >= 2 and <= 5")

    @cached_result
    def ngrams(
        self,
        tokens_table: pl.DataFrame,
        span: int = 2,
        min_frequency: int = 10,
        count_by: str = "pos",
    ) -> pl.DataFrame:
        """
        Generate a table of ngram frequencies of a specified length.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param span: An integer between 2 and 5 representing the size of the ngrams
        :param min_frequency: The minimum count of the ngrams returned
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :return: A polars DataFrame containing token and tag sequences with frequencies
        """  # noqa: E501
        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)
        self._validate_span(span)

        if count_by == "pos":
            grouping_tag = "pos_tag"
            grouping_id = "pos_id"
            expr_filter = pl.col("pos_tag") != "Y"
        else:
            grouping_tag = "ds_tag"
            grouping_id = "ds_id"
            expr_filter = ~(
                pl.col("token").str.contains("^[[[:punct:]] ]+$")
                & pl.col("ds_tag").str.contains("Untagged")
            )

        struct_labels = [f"token_{i}" for i in range(span)]

        look_around_token = [
            pl.col("token").shift(-i).alias(f"tok_lag_{i}") for i in range(span)
        ]
        look_around_tag = [
            pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}") for i in range(span)
        ]

        rename_tokens = [
            pl.col("ngram").struct.rename_fields(
                [f"Token_{i + 1}" for i in range(span)]
            )
        ]
        rename_tags = [
            pl.col("tags").struct.rename_fields([f"Tag_{i + 1}" for i in range(span)])
        ]

        ngram_df = (
            tokens_table.group_by(
                ["doc_id", grouping_id, grouping_tag], maintain_order=True
            )
            .agg(pl.col("token").str.join(""))
            .filter(expr_filter)
            .with_columns(pl.col("token").len().alias("total"))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .with_columns(look_around_token + look_around_tag)
            .group_by("doc_id", "total")
            .agg(
                pl.concat_list([f"tok_lag_{i}" for i in range(span)]).alias("ngram"),
                pl.concat_list([f"tag_lag_{i}" for i in range(span)]).alias("tags"),
            )
            .explode(["ngram", "tags"])
            # Filter out n-grams containing null values (critical fix!)
            .filter(
                pl.col("ngram").list.eval(pl.element().is_not_null()).list.all()
                & pl.col("tags").list.eval(pl.element().is_not_null()).list.all()
            )
            .with_columns(
                pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
            )
            .group_by(["doc_id", "total", "ngram", "tags"])
            .len()
            .sort("len", descending=True)
            .with_columns(pl.struct(["ngram", "tags"]))
            .select(pl.exclude("tags"))
            .pivot(
                index=["ngram", "total"],
                on="doc_id",
                values="len",
                aggregate_function="sum",
            )
            .with_columns(pl.all().exclude("ngram").cast(pl.UInt32, strict=True))
            # calculate range
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("total").is_not_null()
                ).alias("Range")
            )
            .with_columns(pl.selectors.numeric().fill_null(strategy="zero"))
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range")
                .truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude(["Range", "total"]).is_not_null()
                    )
                )
                .mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude(["Range", "total"])
                ).alias("AF")
            )
            .sort("AF", descending=True)
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.col("total")).mul(1000000).alias("RF")
            )
            .select(["ngram", "AF", "RF", "Range"])
            .unnest("ngram")
            .with_columns(rename_tokens + rename_tags)
            .unnest(["ngram", "tags"])
            .sort(
                ["AF", "Token_1", "Token_2"], descending=[True, False, False]
            )  # noqa: E501
            .filter(pl.col("RF") >= min_frequency)
        )

        return ngram_df


class KeynessAnalyzer:
    """Handles keyness analysis and statistical comparisons."""

    @staticmethod
    def _validate_frequency_tables(
        target_frequencies: pl.DataFrame,
        reference_frequencies: pl.DataFrame,
        tags_only: bool = False,
    ) -> None:
        """Validate frequency table schemas."""
        if not tags_only:
            validation = OrderedDict(
                [
                    ("Token", pl.String),
                    ("Tag", pl.String),
                    ("AF", pl.UInt32),
                    ("RF", pl.Float64),
                    ("Range", pl.Float64),
                ]
            )
        else:
            validation = OrderedDict(
                [
                    ("Tag", pl.String),
                    ("AF", pl.UInt32),
                    ("RF", pl.Float64),
                    ("Range", pl.Float64),
                ]
            )

        if (
            target_frequencies.collect_schema() != validation
            or reference_frequencies.collect_schema() != validation
        ):
            table_type = "tags_table" if tags_only else "frequency_table"
            raise ValueError(
                f"Invalid DataFrame. Expected DataFrames produced by {table_type}."  # noqa: E501
            )

    def keyness_table(
        self,
        target_frequencies: pl.DataFrame,
        reference_frequencies: pl.DataFrame,
        correct: bool = False,
        tags_only: bool = False,
        swap_target: bool = False,
        threshold: float = 0.01,
    ) -> pl.DataFrame:
        """
        Generate a keyness table comparing token frequencies from target and reference corpora.

        :param target_frequencies: A frequency table from a target corpus
        :param reference_frequencies: A frequency table from a reference corpus
        :param correct: If True, apply the Yates correction to the log-likelihood calculation
        :param tags_only: If True, assumes frequency tables are from tags_table function
        :param swap_target: If True, swap which corpus is treated as target
        :param threshold: P-value threshold for significance
        :return: A polars DataFrame with keyness statistics
        """  # noqa: E501
        self._validate_frequency_tables(
            target_frequencies, reference_frequencies, tags_only
        )

        total_target = target_frequencies.get_column("AF").sum()
        total_reference = reference_frequencies.get_column("AF").sum()
        total_tokens = total_target + total_reference

        if not correct:
            correction_tar = pl.col("AF")
            correction_ref = pl.col("AF_Ref")
        else:
            correction_tar = pl.col("AF").sub(
                0.5
                * pl.col("AF")
                .sub(
                    (
                        pl.col("AF")
                        .add(pl.col("AF_Ref"))
                        .mul(total_target / total_tokens)
                    )
                )
                .abs()
                .truediv(
                    pl.col("AF").sub(
                        (
                            pl.col("AF")
                            .add(pl.col("AF_Ref"))
                            .mul(total_target / total_tokens)
                        )
                    )
                )
            )
            correction_ref = pl.col("AF_Ref").add(
                0.5
                * pl.col("AF")
                .sub(
                    (
                        pl.col("AF")
                        .add(pl.col("AF_Ref"))
                        .mul(total_target / total_tokens)
                    )
                )
                .abs()
                .truediv(
                    pl.col("AF").sub(
                        (
                            pl.col("AF")
                            .add(pl.col("AF_Ref"))
                            .mul(total_target / total_tokens)
                        )
                    )
                )
            )

        if not tags_only:
            kw_df = target_frequencies.join(
                reference_frequencies,
                on=["Token", "Tag"],
                how="full",
                coalesce=True,
                suffix="_Ref",
            ).fill_null(strategy="zero")
        else:
            kw_df = target_frequencies.join(
                reference_frequencies,
                on="Tag",
                how="full",
                coalesce=True,
                suffix="_Ref",
            ).fill_null(strategy="zero")

        kw_df = (
            kw_df.with_columns(
                pl.when(
                    pl.col("AF")
                    .sub(
                        (
                            pl.col("AF")
                            .add(pl.col("AF_Ref"))
                            .mul(total_target / total_tokens)
                        )
                    )
                    .abs()
                    > 0.25
                )
                .then(correction_tar)
                .otherwise(pl.col("AF"))
                .alias("AF_Yates")
            )
            .with_columns(
                pl.when(
                    pl.col("AF")
                    .sub(
                        (
                            pl.col("AF")
                            .add(pl.col("AF_Ref"))
                            .mul(total_target / total_tokens)
                        )
                    )
                    .abs()
                    > 0.25
                )
                .then(correction_ref)
                .otherwise(pl.col("AF_Ref"))
                .alias("AF_Ref_Yates")
            )
            .with_columns(
                pl.when(pl.col("AF_Yates") > 0)
                .then(
                    pl.col("AF_Yates").mul(
                        pl.col("AF_Yates")
                        .truediv(
                            pl.col("AF_Yates")
                            .add(pl.col("AF_Ref"))
                            .mul(total_target / total_tokens)
                        )
                        .log()
                    )
                )
                .otherwise(0)
                .alias("L1")
            )
            .with_columns(
                pl.when(pl.col("AF_Ref_Yates") > 0)
                .then(
                    pl.col("AF_Ref_Yates").mul(
                        pl.col("AF_Ref_Yates")
                        .truediv(
                            pl.col("AF_Yates")
                            .add(pl.col("AF_Ref_Yates"))
                            .mul(total_reference / total_tokens)
                        )
                        .log()
                    )
                )
                .otherwise(0)
                .alias("L2")
            )
            .with_columns(
                pl.when(pl.col("RF") > pl.col("RF_Ref"))
                .then(pl.col("L1").add(pl.col("L2")).mul(2).abs())
                .otherwise(pl.col("L1").add(pl.col("L2")).mul(2).abs().neg())
                .alias("LL")
            )
            .with_columns(
                pl.when(pl.col("AF_Ref") == 0)
                .then(
                    pl.col("AF")
                    .truediv(total_target)
                    .truediv(0.5 / total_reference)
                    .log(base=2)
                )
                .when(pl.col("AF") == 0)
                .then(
                    pl.col("AF_Ref")
                    .truediv(total_reference)
                    .truediv(0.5 / total_target)
                    .log(base=2)
                    .neg()
                )
                .otherwise(
                    pl.col("AF")
                    .truediv(total_target)
                    .truediv(pl.col("AF_Ref").truediv(total_reference))
                    .log(base=2)
                )
                .alias("LR")
            )
            .with_columns(
                pl.col("LL")
                .abs()
                .map_elements(lambda x: chi2.sf(x, 1), return_dtype=pl.Float64)
                .alias("PV")
            )
            .sort("LL", descending=True)
            .filter(pl.col("PV") < threshold)
        )

        if not swap_target:
            kw_df = kw_df.filter(pl.col("LL") > 0)
        else:
            kw_df = (
                kw_df.with_columns(pl.col(["LL", "LR"]).mul(-1))
                .sort("LL", descending=True)
                .filter(pl.col("LL") > 0)
            )

        if not tags_only:
            return kw_df.select(
                [
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
            )
        else:
            return kw_df.select(
                [
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
            )


class CollocationAnalyzer:
    """Handles collocation analysis and association measures."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    def coll_table(
        self,
        tokens_table: pl.DataFrame,
        node_word: str,
        preceding: int = 4,
        following: int = 4,
        statistic: str = "npmi",
        count_by: str = "pos",
        node_tag: str = None,
    ) -> pl.DataFrame:
        """
        Generate a table of collocations by association measure.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param node_word: The token around which collocations are measured
        :param preceding: Span to the left of the node word (0-9)
        :param following: Span to the right of the node word (0-9)
        :param statistic: Association measure ('pmi', 'npmi', 'pmi2', 'pmi3')
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :param node_tag: Filter node word by tag prefix
        :return: A polars DataFrame with collocation statistics
        """
        stat_types = ["pmi", "npmi", "pmi2", "pmi3"]
        if statistic not in stat_types:
            raise ValueError(f"Invalid statistic type. Expected one of: {stat_types}")

        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)

        if count_by == "pos":
            grouping_tag = "pos_tag"
            grouping_id = "pos_id"
            expr_filter = pl.col("pos_tag") != "Y"
        else:
            grouping_tag = "ds_tag"
            grouping_id = "ds_id"
            expr_filter = ~(
                pl.col("token").str.contains("^[[[:punct:]] ]+$")
                & pl.col("ds_tag").str.contains("Untagged")
            )

        if node_tag is None:
            expr = pl.col("token") == node_word.lower()
        else:
            expr = (pl.col("token") == node_word.lower()) & (
                pl.col(grouping_tag).str.starts_with(node_tag)
            )

        look_around_token = [
            pl.col("token").shift(-i).alias(f"tok_lag_{i}")
            for i in range(-preceding, following + 1)
        ]

        look_around_tag = [
            pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}")
            for i in range(-preceding, following + 1)
        ]

        total_df = (
            tokens_table.group_by(
                ["doc_id", grouping_id, grouping_tag], maintain_order=True
            )
            .agg(pl.col("token").str.join(""))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .filter(expr_filter)
            .group_by(["token", grouping_tag])
            .len(name="Freq_Total")
            .rename({"token": "Token", grouping_tag: "Tag"})
        )

        token_total = sum(total_df.get_column("Freq_Total"))

        if node_tag is None:
            node_freq = (
                total_df.filter(pl.col("Token") == node_word)
                .get_column("Freq_Total")
                .sum()
            )
        else:
            node_freq = (
                total_df.filter(
                    (pl.col("Token") == node_word.lower())
                    & (pl.col("Tag").str.starts_with(node_tag))
                )
                .get_column("Freq_Total")
                .sum()
            )

        if node_freq == 0:
            return pl.DataFrame(
                schema=[
                    ("Token", pl.String),
                    ("Tag", pl.String),
                    ("Freq Span", pl.UInt32),
                    ("Freq Total", pl.UInt32),
                    ("MI", pl.Float64),
                ]
            )

        # Calculate MI function based on statistic type
        if statistic == "pmi":
            mi_funct = (
                pl.col("Freq_Span")
                .truediv(token_total)
                .log(base=2)
                .sub(  # noqa: E501
                    pl.col("Freq_Total")
                    .truediv(token_total)
                    .mul(node_freq)
                    .truediv(token_total)
                    .log(base=2)
                )
            )
        elif statistic == "npmi":
            mi_funct = (
                pl.col("Freq_Span")
                .truediv(token_total)
                .log(base=2)
                .sub(  # noqa: E501
                    pl.col("Freq_Total")
                    .truediv(token_total)
                    .mul(node_freq)
                    .truediv(token_total)
                    .log(base=2)
                )
                .truediv(pl.col("Freq_Span").truediv(token_total).log(base=2).neg())
            )
        elif statistic == "pmi2":
            mi_funct = (
                pl.col("Freq_Span")
                .truediv(token_total)
                .log(base=2)
                .sub(  # noqa: E501
                    pl.col("Freq_Total")
                    .truediv(token_total)
                    .mul(node_freq)
                    .truediv(token_total)
                    .log(base=2)
                )
                .sub(pl.col("Freq_Span").truediv(token_total).log(base=2).mul(-1))
            )
        elif statistic == "pmi3":
            mi_funct = (
                pl.col("Freq_Span")
                .truediv(token_total)
                .log(base=2)
                .sub(  # noqa: E501
                    pl.col("Freq_Total")
                    .truediv(token_total)
                    .mul(node_freq)
                    .truediv(token_total)
                    .log(base=2)
                )
                .sub(pl.col("Freq_Span").truediv(token_total).log(base=2).mul(-2))
            )

        coll_df = (
            tokens_table.group_by(
                ["doc_id", grouping_id, grouping_tag], maintain_order=True
            )
            .agg(pl.col("token").str.join(""))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .filter(pl.col("token").str.contains("[a-z]"))
            .with_columns(look_around_token + look_around_tag)
            .filter(expr)
            .group_by("doc_id")
            .agg(
                pl.concat_list(
                    [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("span_tok"),
                pl.concat_list(
                    [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("span_tag"),
            )
            .explode(["span_tok", "span_tag"])
            .with_columns(
                pre_node_tok=pl.col("span_tok").list.head(preceding),
                pre_node_tag=pl.col("span_tag").list.head(preceding),
            )
            .with_columns(
                post_node_tok=pl.col("span_tok").list.tail(following),
                post_node_tag=pl.col("span_tag").list.tail(following),
            )
            .drop(["span_tok", "span_tag"])
            .with_columns(
                Token=pl.col("pre_node_tok").list.concat("post_node_tok"),
                Tag=pl.col("pre_node_tag").list.concat("post_node_tag"),
            )
            .select(["Token", "Tag"])
            .explode(["Token", "Tag"])
            .group_by(["Token", "Tag"])
            .len(name="Freq_Span")
            .sort("Freq_Span")
            .join(total_df, on=["Token", "Tag"])
            .with_columns(MI=mi_funct)
            .rename({"Freq_Span": "Freq Span", "Freq_Total": "Freq Total"})
            .sort("MI", "Token", descending=[True, False])
        )

        return coll_df


class ClusterAnalyzer:
    """Handles cluster analysis by token and tag."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    @staticmethod
    def _validate_span(span: int) -> None:
        """Validate span parameter."""
        if span < 2 or span > 5:
            raise ValueError("Span must be >= 2 and <= 5")

    def clusters_by_token(
        self,
        tokens_table: pl.DataFrame,
        node_word: str,
        node_position: int = 1,
        span: int = 2,
        search_type: str = "fixed",
        count_by: str = "pos",
    ) -> pl.DataFrame:
        """
        Generate a table of cluster frequencies searching by token.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param node_word: A token to include in the clusters
        :param node_position: Placement of node word in cluster (1=leftmost)
        :param span: Size of clusters (2-5)
        :param search_type: One of 'fixed', 'starts_with', 'ends_with', 'contains'
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :return: A polars DataFrame with cluster frequencies
        """  # noqa: E501
        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)

        if node_position > span:
            node_position = span
            print("Setting node position to right-most position in span.")

        self._validate_span(span)

        if search_type not in [
            "fixed",
            "starts_with",
            "ends_with",
            "contains",
        ]:  # noqa: E501
            raise ValueError(
                "Search type must be one of 'fixed', 'starts_with', "
                "'ends_with', or 'contains'."
            )

        if count_by == "pos":
            grouping_tag = "pos_tag"
            grouping_id = "pos_id"
            expr_filter = pl.col("pos_tag") != "Y"
        else:
            grouping_tag = "ds_tag"
            grouping_id = "ds_id"
            expr_filter = ~(
                pl.col("token").str.contains("^[[[:punct:]] ]+$")
                & pl.col("ds_tag").str.contains("Untagged")
            )

        struct_labels = [f"token_{i}" for i in range(span)]

        if search_type == "fixed":
            expr = pl.col("token") == node_word.lower()
        elif search_type == "starts_with":
            expr = pl.col("token").str.starts_with(node_word.lower())
        elif search_type == "ends_with":
            expr = pl.col("token").str.ends_with(node_word.lower())
        elif search_type == "contains":
            expr = pl.col("token").str.contains(node_word.lower())

        preceding = node_position - 1
        following = span - node_position

        look_around_token = [
            pl.col("token").shift(-i).alias(f"tok_lag_{i}")
            for i in range(-preceding, following + 1)
        ]
        look_around_tag = [
            pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}")
            for i in range(-preceding, following + 1)
        ]

        rename_tokens = [
            pl.col("ngram").struct.rename_fields(
                [f"Token_{i + 1}" for i in range(span)]
            )
        ]
        rename_tags = [
            pl.col("tags").struct.rename_fields([f"Tag_{i + 1}" for i in range(span)])
        ]

        ngram_df = (
            tokens_table.group_by(
                ["doc_id", grouping_id, grouping_tag], maintain_order=True
            )  # noqa: E501
            .agg(pl.col("token").str.join(""))
            .filter(expr_filter)
            .with_columns(pl.col("token").len().alias("total"))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .with_columns(look_around_token + look_around_tag)
            .filter(expr)
            .group_by("doc_id", "total")
            .agg(
                pl.concat_list(
                    [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("ngram"),
                pl.concat_list(
                    [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("tags"),
            )
            .explode(["ngram", "tags"])
        )

        if ngram_df.height == 0:
            return ngram_df

        return (
            ngram_df.with_columns(
                pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
            )
            .group_by(["doc_id", "total", "ngram", "tags"])
            .len()
            .sort("len", descending=True)
            .with_columns(pl.struct(["ngram", "tags"]))
            .select(pl.exclude("tags"))
            .pivot(
                index=["ngram", "total"],
                on="doc_id",
                values="len",
                aggregate_function="sum",
            )
            .with_columns(pl.all().exclude("ngram").cast(pl.UInt32, strict=True))
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("total").is_not_null()
                ).alias("Range")
            )
            .with_columns(pl.selectors.numeric().fill_null(strategy="zero"))
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range")
                .truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude(["Range", "total"]).is_not_null()
                    )
                )
                .mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude(["Range", "total"])
                ).alias("AF")
            )
            .sort("AF", descending=True)
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.col("total")).mul(1000000).alias("RF")
            )
            .select(["ngram", "AF", "RF", "Range"])
            .unnest("ngram")
            .with_columns(rename_tokens + rename_tags)
            .unnest(["ngram", "tags"])
        )

    def clusters_by_tag(
        self,
        tokens_table: pl.DataFrame,
        tag: str,
        tag_position: int = 1,
        span: int = 2,
        count_by: str = "pos",
    ) -> pl.DataFrame:
        """
        Generate a table of cluster frequencies searching by tag.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param tag: A tag to include in the clusters
        :param tag_position: Placement of tag in clusters (1=leftmost)
        :param span: Size of clusters (2-5)
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :return: A polars DataFrame with cluster frequencies
        """
        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)

        if tag_position > span:
            tag_position = span
            print("Setting node position to right-most position in span.")

        self._validate_span(span)

        if count_by == "pos":
            grouping_tag = "pos_tag"
            grouping_id = "pos_id"
            expr = pl.col("pos_tag") == tag
            expr_filter = pl.col("pos_tag") != "Y"
        else:
            grouping_tag = "ds_tag"
            grouping_id = "ds_id"
            expr = pl.col("ds_tag") == tag
            expr_filter = ~(
                pl.col("token").str.contains("^[[[:punct:]] ]+$")
                & pl.col("ds_tag").str.contains("Untagged")
            )

        struct_labels = [f"token_{i}" for i in range(span)]

        preceding = tag_position - 1
        following = span - tag_position

        look_around_token = [
            pl.col("token").shift(-i).alias(f"tok_lag_{i}")
            for i in range(-preceding, following + 1)
        ]
        look_around_tag = [
            pl.col(grouping_tag).shift(-i).alias(f"tag_lag_{i}")
            for i in range(-preceding, following + 1)
        ]

        rename_tokens = [
            pl.col("ngram").struct.rename_fields(
                [f"Token_{i + 1}" for i in range(span)]
            )
        ]
        rename_tags = [
            pl.col("tags").struct.rename_fields([f"Tag_{i + 1}" for i in range(span)])
        ]

        ngram_df = (
            tokens_table.group_by(
                ["doc_id", grouping_id, grouping_tag], maintain_order=True
            )  # noqa: E501
            .agg(pl.col("token").str.join(""))
            .filter(expr_filter)
            .with_columns(pl.col("token").len().alias("total"))
            .with_columns(pl.col("token").str.to_lowercase().str.strip_chars())
            .with_columns(look_around_token + look_around_tag)
            .filter(expr)
            .group_by("doc_id", "total")
            .agg(
                pl.concat_list(
                    [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("ngram"),
                pl.concat_list(
                    [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("tags"),
            )
            .explode(["ngram", "tags"])
        )

        if ngram_df.height == 0:
            return ngram_df

        return (
            ngram_df.with_columns(
                pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
            )
            .group_by(["doc_id", "total", "ngram", "tags"])
            .len()
            .sort("len", descending=True)
            .with_columns(pl.struct(["ngram", "tags"]))
            .select(pl.exclude("tags"))
            .pivot(
                index=["ngram", "total"],
                on="doc_id",
                values="len",
                aggregate_function="sum",
            )
            .with_columns(pl.all().exclude("ngram").cast(pl.UInt32, strict=True))
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("total").is_not_null()
                ).alias("Range")
            )
            .with_columns(pl.selectors.numeric().fill_null(strategy="zero"))
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range")
                .truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude(["Range", "total"]).is_not_null()
                    )
                )
                .mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude(["Range", "total"])
                ).alias("AF")
            )
            .sort("AF", descending=True)
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.col("total")).mul(1000000).alias("RF")
            )
            .select(["ngram", "AF", "RF", "Range"])
            .unnest("ngram")
            .with_columns(rename_tokens + rename_tags)
            .unnest(["ngram", "tags"])
        )


class KWICAnalyzer:
    """Handles Keywords in Context (KWIC) analysis."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    def kwic_center_node(
        self,
        tokens_table: pl.DataFrame,
        node_word: str,
        ignore_case: bool = True,
        search_type: str = "fixed",
    ) -> pl.DataFrame:
        """
        Generate a KWIC table with the node word in the center column.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param node_word: The token of interest
        :param ignore_case: Whether to ignore case in matching
        :param search_type: One of 'fixed', 'starts_with', 'ends_with', 'contains'
        :return: A polars DataFrame with KWIC concordance lines
        """  # noqa: E501
        self._validate_tokens_table(tokens_table)

        if search_type not in [
            "fixed",
            "starts_with",
            "ends_with",
            "contains",
        ]:  # noqa: E501
            raise ValueError(
                "Search type must be one of 'fixed', 'starts_with', "
                "'ends_with', or 'contains'."
            )

        if search_type == "fixed" and ignore_case:
            expr = (
                pl.col("token").str.to_lowercase().str.strip_chars()
                == node_word.lower()
            )
        elif search_type == "fixed" and not ignore_case:
            expr = pl.col("token").str.strip_chars() == node_word
        elif search_type == "starts_with" and ignore_case:
            expr = (
                pl.col("token")
                .str.to_lowercase()
                .str.strip_chars()
                .str.starts_with(node_word.lower())
            )
        elif search_type == "starts_with" and not ignore_case:
            expr = pl.col("token").str.strip_chars().str.starts_with(node_word)
        elif search_type == "ends_with" and ignore_case:
            expr = (
                pl.col("token")
                .str.to_lowercase()
                .str.strip_chars()
                .str.ends_with(node_word.lower())
            )
        elif search_type == "ends_with" and not ignore_case:
            expr = pl.col("token").str.ends_with(node_word)
        elif search_type == "contains" and ignore_case:
            expr = (
                pl.col("token")
                .str.to_lowercase()
                .str.strip_chars()
                .str.contains(node_word.lower())
            )
        elif search_type == "contains" and not ignore_case:
            expr = pl.col("token").str.strip_chars().str.contains(node_word)

        preceding = 7
        following = 7

        look_around_token = [
            pl.col("token").shift(-i).alias(f"tok_lag_{i}")
            for i in range(-preceding, following + 1)
        ]

        kwic_df = (
            tokens_table.group_by(["doc_id", "pos_id"], maintain_order=True)
            .agg(pl.col("token").str.join(""))
            .with_columns(look_around_token)
            .filter(expr)
            .group_by("doc_id")
            .agg(
                pl.concat_list(
                    [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("node")
            )
            .explode("node")
            .with_columns(pre_node=pl.col("node").list.head(7))
            .with_columns(post_node=pl.col("node").list.tail(7))
            .with_columns(pl.col("node").list.get(7))
            .with_columns(pl.col("pre_node").list.join(""))
            .with_columns(pl.col("post_node").list.join(""))
            .select(["doc_id", "pre_node", "node", "post_node"])
            .sort("doc_id")
            .rename(
                {
                    "doc_id": "Doc ID",
                    "pre_node": "Pre-Node",
                    "node": "Node",
                    "post_node": "Post-Node",
                }
            )
        )

        return kwic_df


class DocumentAnalyzer:
    """Handles document-level analysis and tag ruler functionality."""

    @staticmethod
    def _validate_tokens_table(tokens_table: pl.DataFrame) -> None:
        """Validate tokens table schema."""
        validation = OrderedDict(
            [
                ("doc_id", pl.String),
                ("token", pl.String),
                ("pos_tag", pl.String),
                ("ds_tag", pl.String),
                ("pos_id", pl.UInt32),
                ("ds_id", pl.UInt32),
            ]
        )
        if tokens_table.collect_schema() != validation:
            raise ValueError(
                "Invalid DataFrame. "
                "Expected a DataFrame produced by docuscope_parse."
            )

    def tag_ruler(
        self, tokens_table: pl.DataFrame, doc_id: Union[str, int], count_by: str = "pos"
    ) -> pl.DataFrame:
        """
        Retrieve spans of tags to facilitate tag highlighting in a single text.

        :param tokens_table: A polars DataFrame as generated by docuscope_parse
        :param doc_id: A document name or index of a document id
        :param count_by: One of 'pos' or 'ds' for aggregating tokens
        :return: A polars DataFrame with tokens, tags, and span indices
        """
        count_types = ["pos", "ds"]
        if count_by not in count_types:
            raise ValueError(f"Invalid count_by type. Expected one of: {count_types}")

        self._validate_tokens_table(tokens_table)

        max_id = tokens_table.select(pl.col("doc_id").unique().count()).item() - 1

        if isinstance(doc_id, int) and doc_id > max_id:
            raise ValueError(f"Document index {doc_id} exceeds maximum index {max_id}")

        if isinstance(doc_id, int):
            doc = tokens_table.select(pl.col("doc_id").unique().sort()).item(row=doc_id)
        else:
            doc = doc_id

        ruler_df = tokens_table.filter(pl.col("doc_id") == doc)

        if ruler_df.height < 1:
            raise ValueError(f"No data found for document: {doc}")

        if count_by == "pos":
            return ruler_df.select(["token", "pos_tag", "pos_id"]).rename(
                {"pos_tag": "tag", "pos_id": "tag_id"}
            )
        else:
            return ruler_df.select(["token", "ds_tag", "ds_id"]).rename(
                {"ds_tag": "tag", "ds_id": "tag_id"}
            )
