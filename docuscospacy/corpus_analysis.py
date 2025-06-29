"""
Functions for analyzing corpus data tagged with DocuScope and CLAWS7.

This module provides the main API functions for corpus analysis, serving as
convenient wrappers around the modular analyzer classes. These functions
maintain backward compatibility while leveraging the enhanced performance
and validation features of the underlying analyzer classes.

Main Functions:
    docuscope_parse: Parse and tag corpus text using spaCy and DocuScope
    frequency_table: Generate frequency distributions for tokens or tags
    tags_table: Analyze tag frequencies and distributions
    dispersions_table: Calculate token dispersion across documents
    tags_dtm: Create document-term matrices for tags
    ngrams: Extract and analyze n-grams from corpus
    clusters_by_token: Cluster documents by token similarity
    clusters_by_tag: Cluster documents by tag similarity
    kwic_center_node: Generate keywords-in-context concordances
    coll_table: Analyze collocations and word associations
    keyness_table: Compare frequency distributions for keyness analysis
    tag_ruler: Apply rule-based tag modifications

Example:
    Basic corpus analysis workflow::

        import polars as pl
        import spacy
        import docuscospacy as ds

        # Load your corpus
        corpus = pl.DataFrame({
            'doc_id': ['doc1.txt', 'doc2.txt'],
            'text': ['This is the first document.', 'This is the second document.']
        })

        # Load spaCy model with DocuScope tags
        nlp = spacy.load('en_docusco_spacy_lg')

        # Parse and tag the corpus
        tokens = ds.docuscope_parse(corpus, nlp)

        # Generate frequency table
        freq_table = ds.frequency_table(tokens)

        # Analyze tag distributions
        tag_dist = ds.tags_table(tokens)

        # Extract bigrams
        bigrams = ds.ngrams(tokens, n=2)

.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

from typing import Union
import polars as pl
import numpy as np
from spacy.language import Language

from .processors import CorpusProcessor
from .analyzers import (
    FrequencyAnalyzer,
    TagAnalyzer,
    DispersionAnalyzer,
    NGramAnalyzer,
    KeynessAnalyzer,
    CollocationAnalyzer,
    ClusterAnalyzer,
    KWICAnalyzer,
    DocumentAnalyzer,
)

# Initialize analyzer instances for use in wrapper functions
_freq_analyzer = FrequencyAnalyzer()
_tag_analyzer = TagAnalyzer()
_disp_analyzer = DispersionAnalyzer()
_ngram_analyzer = NGramAnalyzer()
_keyness_analyzer = KeynessAnalyzer()
_coll_analyzer = CollocationAnalyzer()
_cluster_analyzer = ClusterAnalyzer()
_kwic_analyzer = KWICAnalyzer()
_doc_analyzer = DocumentAnalyzer()


def _calc_disp2(v: pl.Series, total: float, s=None) -> pl.DataFrame:
    """
    Interate over polars Dataframe to calculate dispersion measures.

    :param v: A polars Series or column of a DataFrame.
    :param total: The total number of tokens in a corpus.
    :return: A polars DataFrame.
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
    # note that this is normalizing according to the normalizing factor 'nf'
    values["RF"] = (f / total) * nf
    values["Carrolls_D2"] = (
        np.log2(f) - (np.sum(v[v != 0] * np.log2(v[v != 0])) / f)
    ) / np.log2(n)
    values["Rosengrens_S"] = np.sum(np.sqrt(v * s)) ** 2 / f
    values["Lynes_D3"] = 1 - (np.sum(((v - np.mean(v)) ** 2) / np.mean(v)) / (4 * f))
    values["DC"] = ((np.sum(np.sqrt(v)) / n) ** 2) / np.mean(v)
    values["Juillands_D"] = 1 - (np.std(v / s) / np.mean(v / s)) / np.sqrt(
        len(v / s) - 1
    )

    values["DP"] = np.sum(np.abs((v / f) - s)) / 2
    # corrected
    values["DP_norm"] = (np.sum(np.abs((v / f) - s)) / 2) / (1 - np.min(s))

    return pl.DataFrame(values)


def docuscope_parse(
    corp: pl.DataFrame, nlp_model: Language, n_process: int = 1, batch_size: int = 25
) -> pl.DataFrame:
    """
    Parse and tag a corpus using the DocuScope spaCy model.

    This is the main entry point for processing raw text through the complete
    DocuScope pipeline. It handles validation, preprocessing, chunking for
    large documents, spaCy processing, and output formatting.

    Args:
        corp: A polars DataFrame containing 'doc_id' and 'text' columns.
            - doc_id: Unique identifier for each document (string)
            - text: Raw text content to be analyzed (string)
        nlp_model: A loaded spaCy Language model instance. Must be an
            'en_docusco_spacy' model with DocuScope components installed.
        n_process: Number of parallel processes for spaCy processing.
            Default is 1 (single-threaded). Higher values can speed up
            processing of large corpora but require more memory.
        batch_size: Number of documents to process in each batch.
            Larger batches are more efficient but use more memory.
            Default is 25.

    Returns:
        A polars DataFrame with the following columns:
            - doc_id: Document identifier (maintains original doc_id values)
            - token: Individual word/token text
            - pos_tag: Part-of-speech tag from CLAWS7 tagset
            - ds_tag: DocuScope tag indicating rhetorical/functional category

    Raises:
        ModelValidationError: If nlp_model is not a compatible DocuScope model.
        CorpusValidationError: If corp doesn't have required columns or format.
        ProcessingError: If processing fails due to memory or other issues.

    Example:
        Basic usage::

            import polars as pl
            import spacy
            import docuscospacy as ds

            # Prepare corpus
            corpus = pl.DataFrame({
                'doc_id': ['doc1.txt', 'doc2.txt'],
                'text': [
                    'This is the first document text.',
                    'This is the second document with different content.'
                ]
            })

            # Load DocuScope model
            nlp = spacy.load('en_docusco_spacy_lg')

            # Process corpus
            tokens = ds.docuscope_parse(corpus, nlp)

            print(tokens.head())
            # doc_id     token    pos_tag    ds_tag
            # doc1.txt   This     DT         B-ConfidenceCertain
            # doc1.txt   is       VBZ        B-Certainty
            # doc1.txt   the      DT         I-ConfidenceCertain

        Parallel processing for large corpora::

            # Use 4 processes for faster processing
            tokens = ds.docuscope_parse(corpus, nlp, n_process=4, batch_size=50)

    Note:
        - Large documents are automatically chunked to prevent memory issues
        - Processing time scales roughly linearly with corpus size
        - The function includes comprehensive validation and error handling
        - Results include performance monitoring for optimization insights
    """
    # Use the new processor pipeline
    processor = CorpusProcessor()
    return processor.process_corpus(corp, nlp_model, n_process, batch_size)


def frequency_table(tokens_table: pl.DataFrame, count_by="pos") -> pl.DataFrame:
    """
    Generate a count of token frequencies.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds', or 'both' for aggregating tokens
    :return: A polars DataFrame of token counts
    """
    return _freq_analyzer.frequency_table(tokens_table, count_by)


def tags_table(tokens_table: pl.DataFrame, count_by="pos") -> pl.DataFrame:
    """
    Generate a count of tag frequencies.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
    :return: a polars DataFrame of absolute frequencies, \
        normalized frequencies(per million tokens) and ranges
    """
    return _tag_analyzer.tags_table(tokens_table, count_by)


def dispersions_table(tokens_table: pl.DataFrame, count_by="pos") -> pl.DataFrame:
    """
    Generate a table of dispersion measures.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a polars DataFrame with various dispersion measures.
    """
    return _disp_analyzer.dispersions_table(tokens_table, count_by)


def tags_dtm(tokens_table: pl.DataFrame, count_by="pos") -> pl.DataFrame:
    """
    Generate a document-term matrix of raw tag counts.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
    :return: a polars DataFrame of absolute tag frequencies for each document
    """
    return _tag_analyzer.tags_dtm(tokens_table, count_by)


def ngrams(
    tokens_table: pl.DataFrame, span=2, min_frequency=10, count_by="pos"
) -> pl.DataFrame:
    """
    Generate a table of ngram frequencies of a specificd length.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param span: An interger between 2 and 5 \
        representing the size of the ngrams
    :param min_frequency: The minimum count of the ngrams returned
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a polars DataFrame containing \
        a token sequence the length of the span, \
            a tag sequence the length of the span, absolute frequencies, \
                normalized frequencies (per million tokens) and ranges
    """
    return _ngram_analyzer.ngrams(tokens_table, span, min_frequency, count_by)


def clusters_by_token(
    tokens_table: pl.DataFrame,
    node_word: str,
    node_position=1,
    span=2,
    search_type="fixed",
    count_by="pos",
):
    """
    Generate a table of cluster frequencies searching by token.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param node_word: A token to include in the clusters
    :param node_position: The placement of the node word in the cluster \
        (1, for example, would be on the left)
    :param span: An interger between 2 and 5 \
        representing the size of the clusters
    :param search_type: One of 'fixed', 'starts_with', \
        'ends_with', or 'contains'
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a polars DataFrame containing \
        a token sequence the length of the span, \
            a tag sequence the length of the span, absolute frequencies, \
                normalized frequencies (per million tokens) and ranges
    """
    return _cluster_analyzer.clusters_by_token(
        tokens_table, node_word, node_position, span, search_type, count_by
    )


def clusters_by_tag(
    tokens_table: pl.DataFrame, tag: str, tag_position=1, span=2, count_by="pos"
) -> pl.DataFrame:
    """
    Generate a table of cluster frequencies searching by tag.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param tag: A tag to include in the clusters
    :param tag_position: The placement of tag in the clusters \
        (1, for example, would be on the left)
    :param span: An interger between 2 and 5 \
        representing the size of the clusters
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a polars DataFrame containing \
        a token sequence the length of the span, \
            a tag sequence the length of the span, absolute frequencies, \
                normalized frequencies (per million tokens) and ranges
    """
    return _cluster_analyzer.clusters_by_tag(
        tokens_table, tag, tag_position, span, count_by
    )


def kwic_center_node(
    tokens_table: pl.DataFrame, node_word: str, ignore_case=True, search_type="fixed"
) -> pl.DataFrame:
    """
    Generate a KWIC table with the node word in the center column.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param node_word: The token of interest
    :param search_type: One of 'fixed', 'starts_with', \
        'ends_with', or 'contains'
    :return: A polars DataFrame containing with the node word \
        in a center column and context columns on either side.
    """
    return _kwic_analyzer.kwic_center_node(
        tokens_table, node_word, ignore_case, search_type
    )


def coll_table(
    tokens_table: pl.DataFrame,
    node_word: str,
    preceding=4,
    following=4,
    statistic="npmi",
    count_by="pos",
    node_tag=None,
) -> pl.DataFrame:
    """
    Generate a table of collocations by association measure.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param node_word: The token around with collocations are measured
    :param preceding: An integer between 0 and 9 \
        representing the span to the left of the node word
    :param following: An integer between 0 and 9 \
        representing the span to the right of the node word
    :param statistic: The association measure to be calculated. \
        One of: 'pmi', 'npmi', 'pmi2', 'pmi3'
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :param node_tag: A value specifying the first character or characters \
        of the node word tag. \
            If the node_word were 'can', a node_tag 'V' \
                would search for can as a verb.
    :return: a polars DataFrame containing collocate tokens, tags, \
        the absolute frequency the collocate in the corpus, \
            the absolute frequency of the collocate within the span, \
                and the association measure.
    """
    return _coll_analyzer.coll_table(
        tokens_table, node_word, preceding, following, statistic, count_by, node_tag
    )


def keyness_table(
    target_frequencies: pl.DataFrame,
    reference_frequencies: pl.DataFrame,
    correct=False,
    tags_only=False,
    swap_target=False,
    threshold=0.01,
):
    """
    Generate a keyness table comparing token frequencies \
        from a taget and a reference corpus

    :param target_frequencies: A frequency table from a target corpus
    :param reference_frequencies: A frequency table from a reference corpus
    :param correct: If True, apply the Yates correction \
        to the log-likelihood calculation
    :param tags_only: If True, it is assumed the frequency tables \
        are of the type produced by the tags_table function
    :return: a polars DataFrame of absolute frequencies, \
        normalized frequencies (per million tokens) \
            and ranges for both corpora, \
                as well as keyness values as calculated by \
                    log-likelihood and effect size as calculated by Log Ratio.
    """
    return _keyness_analyzer.keyness_table(
        target_frequencies,
        reference_frequencies,
        correct,
        tags_only,
        swap_target,
        threshold,
    )


def tag_ruler(
    tokens_table: pl.DataFrame, doc_id: Union[str, int], count_by="pos"
) -> pl.DataFrame:
    """
    Retrieve spans of tags to facilitate tag highligting in a single text.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param doc_id: A document name \
        or an integer representing the index of a document id
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: A polars DataFrame including all tokens, tags, \
        tags start indices, and tag end indices
    """
    return _doc_analyzer.tag_ruler(tokens_table, doc_id, count_by)
