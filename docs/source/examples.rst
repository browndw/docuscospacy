Examples and Usage Guide
========================

This page provides comprehensive examples for using docuscospacy's analyzer
classes and functions.

Basic Corpus Processing
-----------------------

Getting Started
~~~~~~~~~~~~~~~

First, let's set up a basic corpus and process it through the DocuScope
pipeline:

.. code-block:: python

    import polars as pl
    import spacy
    import docuscospacy as ds

    # Create a sample corpus
    corpus = pl.DataFrame({
        'doc_id': ['article1.txt', 'article2.txt', 'article3.txt'],
        'text': [
            'The researchers conducted a comprehensive study on climate change.',
            'This study demonstrates significant findings about global warming.',
            'Climate scientists agree that immediate action is necessary.'
        ]
    })

    # Load the DocuScope spaCy model
    nlp = spacy.load('en_docusco_spacy_lg')

    # Process the corpus
    tokens = ds.docuscope_parse(corpus, nlp)
    print(tokens.head())

Frequency Analysis
------------------

FrequencyAnalyzer Class
~~~~~~~~~~~~~~~~~~~~~~~

The FrequencyAnalyzer provides comprehensive frequency analysis capabilities:

.. code-block:: python

    from docuscospacy.analyzers import FrequencyAnalyzer

    # Initialize analyzer
    freq_analyzer = FrequencyAnalyzer()

    # Basic frequency table
    freq_table = freq_analyzer.frequency_table(tokens)
    print("Top 10 most frequent tokens:")
    print(freq_table.head(10))

    # Filter by minimum frequency
    frequent_tokens = freq_analyzer.frequency_table(tokens, af_min=2)
    print(f"Tokens appearing at least 2 times: {frequent_tokens.height}")

    # Analyze both POS and DocuScope tags
    pos_freq, ds_freq = freq_analyzer.frequency_table(tokens, count_by='both')

    print("Most frequent POS tags:")
    print(pos_freq.head())

    print("Most frequent DocuScope tags:")
    print(ds_freq.head())

Using the functional API:

.. code-block:: python

    # Equivalent using the functional API
    freq_table = ds.frequency_table(tokens)

    # Count by DocuScope tags instead of POS tags
    ds_freq_table = ds.frequency_table(tokens, count_by='ds')

Tag Analysis
------------

TagAnalyzer Class
~~~~~~~~~~~~~~~~~

Analyze tag distributions and patterns:

.. code-block:: python

    from docuscospacy.analyzers import TagAnalyzer

    tag_analyzer = TagAnalyzer()

    # Generate tag frequency table
    tag_freq = tag_analyzer.tags_table(tokens)
    print("Tag distribution:")
    print(tag_freq)

    # Create document-term matrix for tags
    tag_dtm = tag_analyzer.tags_dtm(tokens)
    print(f"DTM shape: {tag_dtm.shape}")
    print("First few columns:", tag_dtm.columns[:5])

Using the functional API:

.. code-block:: python

    # Equivalent functional calls
    tag_freq = ds.tags_table(tokens)
    tag_dtm = ds.tags_dtm(tokens)

N-gram Analysis
---------------

NGramAnalyzer Class
~~~~~~~~~~~~~~~~~~~

Extract and analyze n-grams from your corpus:

.. code-block:: python

    from docuscospacy.analyzers import NGramAnalyzer

    ngram_analyzer = NGramAnalyzer()

    # Extract bigrams (2-grams)
    bigrams = ngram_analyzer.ngrams(tokens, n=2)
    print("Most frequent bigrams:")
    print(bigrams.head(10))

    # Extract trigrams (3-grams)
    trigrams = ngram_analyzer.ngrams(tokens, n=3)
    print("Most frequent trigrams:")
    print(trigrams.head(10))

    # Filter by minimum frequency
    frequent_bigrams = ngram_analyzer.ngrams(tokens, n=2, af_min=2)

Using the functional API:

.. code-block:: python

    # Extract n-grams using functional API
    bigrams = ds.ngrams(tokens, n=2)
    trigrams = ds.ngrams(tokens, n=3)

Dispersion Analysis
-------------------

DispersionAnalyzer Class
~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how tokens are distributed across documents:

.. code-block:: python

    from docuscospacy.analyzers import DispersionAnalyzer

    disp_analyzer = DispersionAnalyzer()

    # Calculate dispersion statistics
    dispersions = disp_analyzer.dispersions_table(tokens)
    print("Token dispersions:")
    print(dispersions.head())

    # Tokens with high dispersion are evenly distributed
    even_tokens = dispersions.filter(pl.col("Dispersion") > 0.8)
    print("Evenly distributed tokens:")
    print(even_tokens)

Using the functional API:

.. code-block:: python

    dispersions = ds.dispersions_table(tokens)

Clustering Analysis
-------------------

ClusterAnalyzer Class
~~~~~~~~~~~~~~~~~~~~~

Cluster documents based on token or tag similarity:

.. code-block:: python

    from docuscospacy.analyzers import ClusterAnalyzer

    cluster_analyzer = ClusterAnalyzer()

    # Cluster by token similarity
    token_clusters = cluster_analyzer.clusters_by_token(tokens, k=2)
    print("Document clusters by token similarity:")
    print(token_clusters)

    # Cluster by tag similarity
    tag_clusters = cluster_analyzer.clusters_by_tag(tokens, k=2)
    print("Document clusters by tag similarity:")
    print(tag_clusters)

Using the functional API:

.. code-block:: python

    token_clusters = ds.clusters_by_token(tokens, k=2)
    tag_clusters = ds.clusters_by_tag(tokens, k=2)

Keywords in Context (KWIC)
--------------------------

KWICAnalyzer Class
~~~~~~~~~~~~~~~~~~

Generate concordances showing keywords in context:

.. code-block:: python

    from docuscospacy.analyzers import KWICAnalyzer

    kwic_analyzer = KWICAnalyzer()

    # Generate KWIC for a specific node word
    kwic_results = kwic_analyzer.kwic_center_node(tokens, node='study')
    print("KWIC results for 'study':")
    print(kwic_results)

    # Customize context window
    kwic_wide = kwic_analyzer.kwic_center_node(tokens, node='climate', span=5)

Using the functional API:

.. code-block:: python

    kwic_results = ds.kwic_center_node(tokens, node='study')

Collocation Analysis
--------------------

CollocationAnalyzer Class
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze word associations and collocations:

.. code-block:: python

    from docuscospacy.analyzers import CollocationAnalyzer

    coll_analyzer = CollocationAnalyzer()

    # Find collocations for a target word
    collocations = coll_analyzer.coll_table(tokens, target='climate')
    print("Collocations for 'climate':")
    print(collocations.head())

    # Adjust context window
    close_collocations = coll_analyzer.coll_table(tokens, target='study', span=2)

Using the functional API:

.. code-block:: python

    collocations = ds.coll_table(tokens, target='climate')

Keyness Analysis
----------------

KeynessAnalyzer Class
~~~~~~~~~~~~~~~~~~~~~

Compare frequency distributions between corpora for keyness:

.. code-block:: python

    from docuscospacy.analyzers import KeynessAnalyzer

    # Process a reference corpus
    reference_corpus = pl.DataFrame({
        'doc_id': ['ref1.txt', 'ref2.txt'],
        'text': [
            'This is a reference document about different topics.',
            'Reference texts provide baseline comparisons for analysis.'
        ]
    })

    reference_tokens = ds.docuscope_parse(reference_corpus, nlp)

    # Generate frequency tables
    target_freq = ds.frequency_table(tokens)
    reference_freq = ds.frequency_table(reference_tokens)

    # Calculate keyness
    keyness_analyzer = KeynessAnalyzer()
    keyness_results = keyness_analyzer.keyness_table(target_freq, reference_freq)
    print("Keyness analysis results:")
    print(keyness_results.head())

Using the functional API:

.. code-block:: python

    keyness_results = ds.keyness_table(target_freq, reference_freq)

Performance and Caching
------------------------

Using Performance Features
~~~~~~~~~~~~~~~~~~~~~~~~~~

All analyzer classes include automatic caching and performance monitoring:

.. code-block:: python

    from docuscospacy.performance import PerformanceCache, PerformanceMonitor

    # Performance is automatically monitored
    with PerformanceMonitor("My analysis"):
        freq_table = freq_analyzer.frequency_table(tokens)
        tag_table = tag_analyzer.tags_table(tokens)

    # Results are automatically cached - subsequent calls are faster
    # This call will use cached results
    freq_table_cached = freq_analyzer.frequency_table(tokens)

    # Clear cache if needed
    cache = PerformanceCache()
    cache.clear_cache()

Memory Optimization
~~~~~~~~~~~~~~~~~~~

For large corpora, use memory optimization features:

.. code-block:: python

    from docuscospacy.performance import MemoryOptimizer

    # Automatically optimize memory usage
    optimizer = MemoryOptimizer()

    # Check if corpus is large
    if optimizer.is_large_corpus(tokens):
        print("Large corpus detected - using memory optimizations")

    # Process in batches for large corpora
    with optimizer.batch_processing(tokens, batch_size=1000) as batches:
        results = []
        for batch in batches:
            batch_result = freq_analyzer.frequency_table(batch)
            results.append(batch_result)

Error Handling and Validation
------------------------------

Robust Error Handling
~~~~~~~~~~~~~~~~~~~~~~

Use comprehensive validation and error handling:

.. code-block:: python

    from docuscospacy.validation import (
        CorpusValidationError, ModelValidationError, validate_corpus_dataframe
    )

    # Validate corpus before processing
    try:
        validate_corpus_dataframe(corpus)
        print("Corpus validation passed!")
    except CorpusValidationError as e:
        print(f"Corpus validation failed: {e}")
        # The error message includes suggestions for fixing the issue

    # Handle model validation
    try:
        tokens = ds.docuscope_parse(corpus, nlp)
    except ModelValidationError as e:
        print(f"Model validation failed: {e}")
        # Error includes link to correct model download

    # Catch all docuscospacy errors
    from docuscospacy.validation import DocuscoSpacyError

    try:
        # Your analysis code
        results = ds.frequency_table(tokens)
    except DocuscoSpacyError as e:
        print(f"Analysis failed: {e}")

Advanced Workflows
------------------

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's an example of a complete analysis workflow:

.. code-block:: python

    import polars as pl
    import spacy
    import docuscospacy as ds
    from docuscospacy.analyzers import *

    def complete_corpus_analysis(corpus_df, nlp_model):
        """Complete corpus analysis workflow."""

        # Step 1: Process corpus
        print("Processing corpus...")
        tokens = ds.docuscope_parse(corpus_df, nlp_model)

        # Step 2: Basic frequency analysis
        print("Analyzing frequencies...")
        freq_table = ds.frequency_table(tokens)
        tag_table = ds.tags_table(tokens)

        # Step 3: N-gram analysis
        print("Extracting n-grams...")
        bigrams = ds.ngrams(tokens, n=2)
        trigrams = ds.ngrams(tokens, n=3)

        # Step 4: Dispersion analysis
        print("Calculating dispersions...")
        dispersions = ds.dispersions_table(tokens)

        # Step 5: Document clustering
        print("Clustering documents...")
        clusters = ds.clusters_by_token(tokens, k=3)

        # Step 6: Create DTM for further analysis
        print("Creating document-term matrix...")
        dtm = ds.tags_dtm(tokens)

        # Return comprehensive results
        return {
            'tokens': tokens,
            'frequencies': freq_table,
            'tags': tag_table,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'dispersions': dispersions,
            'clusters': clusters,
            'dtm': dtm
        }

    # Run complete analysis
    corpus = pl.DataFrame({
        'doc_id': ['doc1.txt', 'doc2.txt', 'doc3.txt'],
        'text': ['Your document texts here...'] * 3
    })

    nlp = spacy.load('en_docusco_spacy_lg')
    results = complete_corpus_analysis(corpus, nlp)

    print(f"Analysis complete! Generated {len(results)} result tables.")

Working with Large Corpora
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tips for processing large corpora efficiently:

.. code-block:: python

    # For very large corpora, use parallel processing
    large_tokens = ds.docuscope_parse(
        large_corpus,
        nlp,
        n_process=4,  # Use multiple processes
        batch_size=100  # Larger batches
    )

    # Use memory-efficient analysis
    from docuscospacy.performance import MemoryOptimizer

    optimizer = MemoryOptimizer()

    # Process frequency analysis in batches
    if optimizer.is_large_corpus(large_tokens):
        print("Using memory-optimized processing...")

        # Analyze in chunks
        chunk_size = 10000
        total_rows = large_tokens.height

        freq_results = []
        for i in range(0, total_rows, chunk_size):
            chunk = large_tokens.slice(i, chunk_size)
            chunk_freq = ds.frequency_table(chunk)
            freq_results.append(chunk_freq)

        # Combine results
        combined_freq = pl.concat(freq_results)
        final_freq = combined_freq.group_by(['Token', 'Tag']).agg([
            pl.col('AF').sum(),
            pl.col('RF').mean(),
            pl.col('Range').mean()
        ])

This comprehensive guide should help users understand how to effectively use all
the analyzer classes and functions in docuscospacy!
