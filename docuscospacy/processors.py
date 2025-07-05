"""
Core processing classes for corpus analysis.

This module provides a complete processing pipeline for text corpora, from
validation and preprocessing through spaCy NLP processing to final output
formatting. The classes are designed to work together as a cohesive pipeline
while also being usable independently.

Classes:
    CorpusValidator: Validates corpus data and spaCy models
    TextPreprocessor: Handles text cleaning and preprocessing
    TextChunker: Splits large texts into manageable chunks
    SpacyProcessor: Processes text through spaCy NLP pipeline
    DataFrameTransformer: Transforms and formats output DataFrames
    CorpusProcessor: Main orchestrator for the complete pipeline

Example:
    Basic usage with the main processor::

        import polars as pl
        import spacy
        from docuscospacy.processors import CorpusProcessor

        # Load your corpus
        corpus = pl.DataFrame({
            'doc_id': ['doc1.txt', 'doc2.txt'],
            'text': ['First document.', 'Second document.']
        })

        # Load spaCy model
        nlp = spacy.load('en_docusco_spacy_lg')

        # Process corpus
        processor = CorpusProcessor()
        tokens = processor.process_corpus(corpus, nlp)

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

import math
import unicodedata
from typing import List, Dict, Any, Tuple

import polars as pl
from spacy.tokens import Doc
from spacy.language import Language

from .config import CONFIG, PATTERNS
from .performance import PerformanceMonitor, ProgressTracker, MemoryOptimizer
from .validation import validate_corpus_dataframe, ModelValidationError


# Remove the duplicate classes since they're now in validation.py


class CorpusValidator:
    """
    Validates corpus data and spaCy models for compatibility.

    This class provides static methods for validating that input data
    conforms to expected schemas and that spaCy models are compatible
    with the docuscospacy processing pipeline.

    Example:
        Validate a corpus before processing::

            import polars as pl
            from docuscospacy.processors import CorpusValidator

            corpus = pl.DataFrame({
                'doc_id': ['doc1.txt', 'doc2.txt'],
                'text': ['First document.', 'Second document.']
            })

            # This will raise an exception if validation fails
            CorpusValidator.validate_corpus_schema(corpus)

        Validate a spaCy model::

            import spacy

            nlp = spacy.load('en_docusco_spacy_lg')
            CorpusValidator.validate_spacy_model(nlp)
    """

    @staticmethod
    def validate_corpus_schema(corp: pl.DataFrame) -> None:
        """
        Validate that corpus has expected schema.

        Checks that the DataFrame contains the required 'doc_id' and 'text'
        columns and meets minimum data quality requirements.

        Args:
            corp: A polars DataFrame that should contain 'doc_id' and 'text' columns.

        Raises:
            CorpusValidationError: If the corpus doesn't meet validation requirements.

        Example:
            >>> import polars as pl
            >>> corpus = pl.DataFrame({'doc_id': ['doc1'], 'text': ['Hello world']})
            >>> CorpusValidator.validate_corpus_schema(corpus)  # No exception raised
        """
        validate_corpus_dataframe(corp, "in CorpusProcessor")

    @staticmethod
    def validate_tokens_schema(tokens_table: pl.DataFrame) -> None:
        """
        Validate that tokens table has expected schema.

        Checks that the DataFrame contains the required columns for processed
        tokens including 'doc_id', 'token', 'pos_tag', and 'ds_tag'.

        Args:
            tokens_table: A polars DataFrame containing processed tokens.

        Raises:
            ValidationError: If the tokens table doesn't meet validation requirements.

        Example:
            >>> import polars as pl
            >>> tokens = pl.DataFrame({
            ...     'doc_id': ['doc1'] * 3,
            ...     'token': ['Hello', 'world', '.'],
            ...     'pos_tag': ['UH', 'NN', 'Y'],
            ...     'ds_tag': ['B-Greeting', 'B-Object', 'O-']
            ... })
            >>> CorpusValidator.validate_tokens_schema(tokens)  # No exception raised
        """
        from .validation import validate_tokens_dataframe

        validate_tokens_dataframe(tokens_table, "in CorpusProcessor")

    @staticmethod
    def validate_spacy_model(nlp_model: Language) -> None:
        """
        Validate that spaCy model is compatible.

        Checks that the provided spaCy model is a docusco_spacy model which
        contains the necessary custom components for DocuScope analysis.

        Args:
            nlp_model: A spaCy Language model instance.

        Raises:
            ModelValidationError: If the model is not a compatible docusco_spacy model.

        Example:
            >>> import spacy
            >>> nlp = spacy.load('en_docusco_spacy_lg')
            >>> CorpusValidator.validate_spacy_model(nlp)  # No exception raised

            >>> invalid_nlp = spacy.blank('en')
            >>> CorpusValidator.validate_spacy_model(invalid_nlp)  # Raises exception
        """
        model_name = nlp_model.lang + "_" + nlp_model.meta["name"]
        if not model_name.startswith(CONFIG.EXPECTED_MODEL_PREFIX):
            raise ModelValidationError(
                f"Invalid spaCy model. Expected a model starting with "
                f"'{CONFIG.EXPECTED_MODEL_PREFIX}'. "
                "For information and instructions see: "
                "https://huggingface.co/browndw/en_docusco_spacy"
            )


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing for NLP analysis.

    This class provides various text normalization and cleaning methods
    to prepare raw text for spaCy processing. All methods can be used
    independently or combined through the main preprocessing pipeline.

    Example:
        Use individual preprocessing methods::

            from docuscospacy.processors import TextPreprocessor

            preprocessor = TextPreprocessor()

            # Clean whitespace
            clean_text = preprocessor.squish_whitespace("Hello    world\\n\\n")
            # Result: "Hello world"

            # Fix curly quotes
            fixed_quotes = preprocessor.replace_curly_quotes("He said "hello"")
            # Result: 'He said "hello"'

        Use the complete preprocessing pipeline::

            import polars as pl

            corpus = pl.DataFrame({
                'doc_id': ['doc1'],
                'text': ['He said "hello"    with   extra spaces']
            })

            preprocessor = TextPreprocessor()
            clean_corpus = preprocessor.preprocess_corpus(corpus)
    """

    @staticmethod
    def squish_whitespace(text: str) -> str:
        """
        Remove extra spaces, returns, tabs, and other whitespace from text.

        Normalizes all whitespace sequences to single spaces and strips
        leading/trailing whitespace.

        Args:
            text: Input text that may contain irregular whitespace.

        Returns:
            Cleaned text with normalized whitespace.

        Example:
            >>> TextPreprocessor.squish_whitespace("Hello    world\\n\\ttest")
            'Hello world test'
        """
        return " ".join(text.split())

    @staticmethod
    def replace_curly_quotes(text: str) -> str:
        """
        Replace curly/smart quotes with straight ASCII quotes.

        Converts Unicode curly quotes (left and right, single and double)
        to standard ASCII quote characters for consistent processing.

        Args:
            text: Input text that may contain curly quotes.

        Returns:
            Text with curly quotes replaced by straight quotes.

        Example:
            >>> TextPreprocessor.replace_curly_quotes("He said "hello"")
            'He said "hello"'
        """
        replacements = {
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201C": '"',  # Left double quote
            "\u201D": '"',  # Right double quote
        }

        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters and convert to ASCII when possible.

        Applies NFKD (compatibility decomposition) normalization and
        converts to ASCII, removing characters that can't be represented.
        This helps standardize text from different sources.

        Args:
            text: Input text that may contain non-ASCII unicode characters.

        Returns:
            ASCII-normalized text.

        Example:
            >>> TextPreprocessor.normalize_unicode("café naïve")
            'cafe naive'
        """
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("utf-8")
        )

    def preprocess_corpus(self, corp: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all preprocessing steps to a corpus DataFrame.

        Runs the complete preprocessing pipeline on the 'text' column
        of a corpus DataFrame, applying all cleaning and normalization
        steps in the optimal order.

        Args:
            corp: A polars DataFrame with 'doc_id' and 'text' columns.

        Returns:
            DataFrame with preprocessed text column.

        Example:
            >>> import polars as pl
            >>> corpus = pl.DataFrame({
            ...     'doc_id': ['doc1'],
            ...     'text': ['He said "hello"    with   extra spaces']
            ... })
            >>> preprocessor = TextPreprocessor()
            >>> clean_corpus = preprocessor.preprocess_corpus(corpus)
        """
        return (
            corp.with_columns(
                pl.col("text").map_elements(
                    self.squish_whitespace, return_dtype=pl.String
                )
            )
            .with_columns(
                pl.col("text").map_elements(
                    self.replace_curly_quotes, return_dtype=pl.String
                )
            )
            .with_columns(
                pl.col("text").map_elements(
                    self.normalize_unicode, return_dtype=pl.String
                )
            )
            .with_columns(
                pl.col("text").str.replace_all(PATTERNS.ITS_PATTERN_STR, r"${1} ${2}")
            )
        )


class TextChunker:
    """
    Handles splitting large texts into manageable chunks for processing.

    This class provides intelligent text chunking that attempts to split
    on natural boundaries (sentences, then words) to preserve linguistic
    context while keeping chunks within memory limits for spaCy processing.

    Example:
        Split a long document::

            from docuscospacy.processors import TextChunker

            chunker = TextChunker()
            long_text = "First sentence. Second sentence. " * 1000

            # Split into 3 chunks
            chunks = chunker.split_document(long_text, 3)
            print(f"Split into {len(chunks)} chunks")

        Process a corpus with automatic chunking::

            import polars as pl

            corpus = pl.DataFrame({
                'doc_id': ['long_doc.txt'],
                'text': ['Very long document text...']
            })

            chunker = TextChunker()
            chunked_corpus = chunker.prepare_chunked_corpus(corpus)
    """

    @staticmethod
    def split_document(doc_txt: str, n_chunks: float) -> List[str]:
        """
        Split documents into chunks, preferring natural boundaries.

        Attempts to split on sentence boundaries first, then word boundaries,
        and finally character boundaries as fallbacks. This preserves
        linguistic context while ensuring manageable chunk sizes.

        Args:
            doc_txt: The document text to be split.
            n_chunks: Number of chunks to create (can be fractional).

        Returns:
            List of text chunks, ideally split on natural boundaries.

        Example:
            >>> chunker = TextChunker()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> chunks = chunker.split_document(text, 2)
            >>> len(chunks)
            2

        Note:
            If n_chunks <= 1, returns the original text as a single chunk.
        """
        if n_chunks <= 0:
            return []  # Return empty list for empty documents
        if n_chunks <= 1:
            return [doc_txt]  # Return single chunk for short documents

        doc_len = len(doc_txt)
        chunk_idx = [math.ceil(i / n_chunks * doc_len) for i in range(1, int(n_chunks))]

        # Try to split on sentence boundaries first
        try:
            split_idx = [
                (PATTERNS.SENTENCE_BOUNDARY.search(doc_txt[idx:]).span()[1] + (idx - 1))
                for idx in chunk_idx
            ]
            split_idx.insert(0, 0)
            doc_chunks = [
                doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:] + [None])
            ]

            if len(doc_chunks) == n_chunks:
                return doc_chunks
        except (AttributeError, IndexError):
            # Fall back to word boundaries if sentence splitting fails
            pass

        # Fallback to word boundaries
        try:
            split_idx = [
                PATTERNS.WORD_BOUNDARY.search(doc_txt[idx:]).span()[0] + idx
                for idx in chunk_idx
            ]
            split_idx.insert(0, 0)
            doc_chunks = [
                doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:] + [None])
            ]
            return doc_chunks
        except (AttributeError, IndexError):
            # If all else fails, just split at character boundaries
            split_idx = chunk_idx[:]
            split_idx.insert(0, 0)
            split_idx.append(len(doc_txt))
            return [doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:])]

    def prepare_chunked_corpus(self, corp: pl.DataFrame) -> pl.DataFrame:
        """
        Split long texts into chunks and prepare for processing.

        Automatically determines which documents need chunking based on
        their length, splits them appropriately, and creates a new corpus
        DataFrame with chunked documents that have unique identifiers.

        Args:
            corp: A polars DataFrame with 'doc_id' and 'text' columns.

        Returns:
            DataFrame with potentially more rows (due to chunking) where
            each chunk has a unique doc_id formed by combining chunk number
            and original doc_id.

        Example:
            >>> import polars as pl
            >>> corpus = pl.DataFrame({
            ...     'doc_id': ['doc1.txt', 'very_long_doc.txt'],
            ...     'text': ['Short text', 'Very long text that needs chunking...']
            ... })
            >>> chunker = TextChunker()
            >>> chunked = chunker.prepare_chunked_corpus(corpus)
            # May result in doc_ids like: 'doc1.txt', '0::very_long_doc.txt', '1::very_long_doc.txt'
        """  # noqa: E501
        return (
            corp.with_columns(
                n_chunks=pl.Expr.ceil(
                    pl.col("text").str.len_chars().truediv(CONFIG.MAX_CHUNK_SIZE)
                ).cast(pl.UInt32, strict=False)
            )
            .with_columns(chunk_id=pl.int_ranges("n_chunks"))
            .with_columns(
                pl.struct(["text", "n_chunks"])
                .map_elements(
                    lambda x: self.split_document(x["text"], x["n_chunks"]),
                    return_dtype=pl.List(pl.String),
                )
                .alias("text")
            )
            .explode("text", "chunk_id")
            .filter(pl.col("text").is_not_null())
            .with_columns(pl.col("text").str.strip_chars() + " ")
            .with_columns(
                pl.concat_str(
                    [pl.col("chunk_id"), pl.col("doc_id")],
                    separator=CONFIG.CHUNK_ID_SEPARATOR,
                ).alias("doc_id")
            )
            .drop(["n_chunks", "chunk_id"])
        )


class SpacyProcessor:
    """Handles spaCy NLP pipeline processing."""

    @staticmethod
    def prepare_text_tuples(corp: pl.DataFrame) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert corpus DataFrame to spaCy-compatible tuple format."""
        text_tuples = []
        for item in corp.to_dicts():
            text_tuples.append((item["text"], {"doc_id": item["doc_id"]}))
        return text_tuples

    @staticmethod
    def setup_doc_extension() -> None:
        """Setup spaCy Doc extension for document IDs."""
        if not Doc.has_extension("doc_id"):
            Doc.set_extension("doc_id", default=None)

    def process_with_spacy(
        self,
        text_tuples: List[Tuple[str, Dict[str, Any]]],
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> List[pl.DataFrame]:
        """Process text tuples through spaCy pipeline."""
        self.setup_doc_extension()

        if show_progress:
            progress = ProgressTracker(len(text_tuples), "spaCy processing")

        doc_tuples = nlp_model.pipe(
            text_tuples, as_tuples=True, n_process=n_process, batch_size=batch_size
        )

        df_list = []
        processed_count = 0

        for doc, context in doc_tuples:
            doc._.doc_id = context["doc_id"]

            # Extract token information
            token_list = [token.text for token in doc]
            ws_list = [token.whitespace_ for token in doc]
            tag_list = [token.tag_ for token in doc]
            iob_list = [token.ent_iob_ for token in doc]
            ent_list = [token.ent_type_ for token in doc]

            # Combine IOB and entity tags
            iob_ent = list(map("-".join, zip(iob_list, ent_list)))

            df = pl.DataFrame(
                {
                    "doc_id": doc._.doc_id,
                    "token": token_list,
                    "ws": ws_list,
                    "pos_tag": tag_list,
                    "ds_tag": iob_ent,
                }
            )
            df_list.append(df)

            processed_count += 1
            if show_progress:
                progress.update()

        if show_progress:
            progress.finish()

        return df_list


class DataFrameTransformer:
    """Handles DataFrame transformations and formatting."""

    @staticmethod
    def recombine_chunks(
        df: pl.DataFrame, original_doc_order: List[str] = None
    ) -> pl.DataFrame:
        """Recombine chunked documents and preserve original document order."""
        # Split chunk info from doc_id
        df_split = (
            df.with_columns(
                pl.col("doc_id").str.split_exact(CONFIG.CHUNK_ID_SEPARATOR, 1)
            )
            .unnest("doc_id")
            .rename({"field_0": "chunk_id", "field_1": "doc_id"})
            .with_columns(pl.col("chunk_id").cast(pl.UInt32, strict=False))
        )

        # If we have original document order, preserve it
        if original_doc_order is not None:
            # Create order mapping
            doc_order_map = {doc_id: i for i, doc_id in enumerate(original_doc_order)}

            # Add order column and sort by it, then by chunk_id
            df_with_order = df_split.with_columns(
                pl.col("doc_id").map_elements(
                    lambda x: doc_order_map.get(x, 999999),  # Unknown docs go to end
                    return_dtype=pl.UInt32
                ).alias("doc_order")
            )

            result = (
                df_with_order
                .sort(["doc_order", "chunk_id"], descending=[False, False])
                .drop(["chunk_id", "doc_order"])
            )
        else:
            # Fallback: sort by doc_id and chunk_id (preserves alphabetical order)
            result = (
                df_split
                .sort(["doc_id", "chunk_id"], descending=[False, False])
                .drop("chunk_id")
            )

        return result

    @staticmethod
    def add_pos_ids(df: pl.DataFrame) -> pl.DataFrame:
        """Add unique IDs to part-of-speech tags for grouping."""
        return (
            df.with_columns(
                pl.when(
                    pl.col("pos_tag").str.contains(r"\\d\\d$")
                    & pl.col("pos_tag").str.contains(r"[^1]$")
                )
                .then(0)
                .otherwise(1)
                .cast(pl.UInt32, strict=False)
                .alias("pos_id")
            )
            .with_columns(
                pl.when(pl.col("pos_id") == 1)
                .then(pl.cum_sum("pos_id"))
                .otherwise(None)
                .forward_fill()
            )
            .with_columns(
                pl.when(
                    pl.col("pos_tag").str.contains(r"\\d\\d$")
                    & pl.col("pos_tag").str.contains(r"[^1]$")
                )
                .then(None)
                .otherwise(pl.col("pos_tag").str.replace(r"\\d\\d$", ""))
                .forward_fill()
                .name.keep()
            )
        )

    @staticmethod
    def add_ds_ids(df: pl.DataFrame) -> pl.DataFrame:
        """Add unique IDs to DocuScope tags for grouping."""
        return (
            df.with_columns(
                pl.when(
                    pl.col("ds_tag").str.starts_with("B-")
                    | pl.col("ds_tag").str.starts_with("O-")
                )
                .then(1)
                .otherwise(0)
                .cast(pl.UInt32, strict=False)
                .alias("ds_id")
            )
            .with_columns(
                pl.when(pl.col("ds_id") == 1)
                .then(pl.cum_sum("ds_id"))
                .otherwise(None)
                .forward_fill()
            )
            .with_columns(
                pl.when(
                    pl.col("ds_tag").str.starts_with("B-")
                    | pl.col("ds_tag").str.starts_with("O-")
                )
                .then(pl.col("ds_tag").str.strip_chars_start("B-"))
                .otherwise(None)
                .forward_fill()
            )
            .with_columns(
                pl.when(pl.col("ds_tag") == "O-")
                .then(pl.col("ds_tag").str.replace("O-", "Untagged"))
                .otherwise(pl.col("ds_tag"))
            )
        )

    @staticmethod
    def apply_tag_corrections(df: pl.DataFrame) -> pl.DataFrame:
        """Apply manual tag corrections and rules."""
        return (
            df.with_columns(
                pl.when(pl.col("token").str.contains(r"^[[:punct:]]+$"))
                .then(pl.lit("Y").alias("pos_tag"))
                .otherwise(pl.col("pos_tag"))
            )
            .with_columns(pl.col("token").shift(1).alias("token_1"))
            .with_columns(
                pl.when(
                    (pl.col("token").str.contains(r"(?i)^s$"))
                    & (pl.col("token_1").str.contains(r"(?i)^it$"))
                )
                .then(pl.lit("GE").alias("pos_tag"))
                .otherwise(pl.col("pos_tag"))
            )
        )

    @staticmethod
    def finalize_tokens(df: pl.DataFrame) -> pl.DataFrame:
        """Combine tokens with whitespace and clean up."""
        return df.with_columns(
            pl.concat_str([pl.col("token"), pl.col("ws")], separator="").alias("token")
        ).drop(["token_1", "ws"])

    def transform_spacy_output(
        self, df_list: List[pl.DataFrame], original_doc_order: List[str] = None
    ) -> pl.DataFrame:
        """Apply all transformations to spaCy output."""
        # Concatenate all DataFrames
        df = pl.concat(df_list)

        # Apply transformations in sequence
        df = self.recombine_chunks(df, original_doc_order=original_doc_order)
        df = self.add_pos_ids(df)
        df = self.add_ds_ids(df)
        df = self.apply_tag_corrections(df)
        df = self.finalize_tokens(df)

        return df


class CorpusProcessor:
    """Main class that orchestrates corpus processing pipeline."""

    def __init__(self):
        self.validator = CorpusValidator()
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.spacy_processor = SpacyProcessor()
        self.transformer = DataFrameTransformer()

    def process_corpus(
        self,
        corp: pl.DataFrame,
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        show_progress: bool = None,
    ) -> pl.DataFrame:
        """
        Process a corpus using the complete pipeline.

        :param corp: A polars DataFrame containing 'doc_id' and 'text' columns.
        :param nlp_model: An 'en_docusco_spacy' instance.
        :param n_process: The number of parallel processes to use
                         during parsing.
        :param batch_size: The batch size to use during parsing.
        :param show_progress: Whether to show progress for large corpora.
                             If None, will auto-determine based on corpus size.
        :return: A polars DataFrame with token sequences identified by both
                 part-of-speech tags and DocuScope tags.
        """
        with PerformanceMonitor("Corpus processing"):
            # Validate inputs
            self.validator.validate_corpus_schema(corp)
            self.validator.validate_spacy_model(nlp_model)

            # Filter out empty texts
            corp = corp.filter(pl.col("text").is_not_null())

            # Capture original document order before any transformations
            original_doc_order = corp["doc_id"].to_list()

            # Determine if we should show progress
            if show_progress is None:
                show_progress = len(corp) > CONFIG.PROGRESS_THRESHOLD

            # Initialize progress tracker
            if show_progress:
                progress = ProgressTracker(len(corp), "Processing corpus")

            # Check if this is a large corpus that needs memory optimization
            if MemoryOptimizer.is_large_corpus_size(len(corp)):
                # Enable memory efficient mode temporarily
                original_mode = CONFIG.MEMORY_EFFICIENT_MODE
                CONFIG.MEMORY_EFFICIENT_MODE = True

            try:
                # Preprocess text
                corp = self.preprocessor.preprocess_corpus(corp)
                if show_progress:
                    progress.update(len(corp) // 4)

                # Chunk large texts
                corp = self.chunker.prepare_chunked_corpus(corp)
                if show_progress:
                    progress.update(len(corp) // 4)

                # Prepare for spaCy processing
                text_tuples = self.spacy_processor.prepare_text_tuples(corp)

                # Process with spaCy
                df_list = self.spacy_processor.process_with_spacy(
                    text_tuples,
                    nlp_model,
                    n_process,
                    batch_size,
                    show_progress=show_progress,
                )
                if show_progress:
                    progress.update(len(corp) // 4)

                # Transform and finalize
                result = self.transformer.transform_spacy_output(
                    df_list, original_doc_order=original_doc_order
                )
                if show_progress:
                    progress.update(len(corp) // 4)
                    progress.finish()

                # Optimize result DataFrame
                result = MemoryOptimizer.optimize_dataframe(result)

                return result

            finally:
                # Restore original memory mode if we changed it
                if MemoryOptimizer.is_large_corpus_size(len(corp)):
                    CONFIG.MEMORY_EFFICIENT_MODE = original_mode
