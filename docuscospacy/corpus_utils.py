"""
Misc. utility functions for corpus processing.

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

import os
import warnings
import polars as pl
from typing import List, OrderedDict
from pathlib import Path
from scipy.sparse import coo_matrix


def get_text_paths(directory: str,
                   recursive=False) -> List:
    """
    Gets a list of full paths for all files \
        and directories in the given directory.

    :param directory: A string represting a path to directory.
    :param recursive: Whether or not to \
        recursively search through subdirectories.
    :return: A list of paths to plain text (TXT) files.
    """
    full_paths = []
    if recursive is True:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    full_paths.append(os.path.join(root, file))
    else:
        for file in Path(directory).glob("*.txt"):
            full_paths.append(str(file))
    return full_paths


def readtext(paths: List) -> pl.DataFrame:
    """
    Read in text (TXT) files from a list of paths \
        into a polars DataFrame with 'doc_id' and 'text' columns.

    :param paths: A list of strings representing \
        paths to plain text (TXT) files.
    :return: A polars DataFrame with 'doc_id' and 'text' columns.
    """
    # Get a list of the file basenames
    doc_ids = [os.path.basename(path) for path in paths]
    # Create a list collapsing each text file into one element in a string
    texts = [open(path).read() for path in paths]
    df = pl.DataFrame({
        "doc_id": doc_ids,
        "text": texts
    })
    df = (
        df
        .with_columns(
            pl.col("text").str.strip_chars()
        )
        .sort("doc_id", descending=False)
    )
    return df


def corpus_from_folder(directory: str) -> pl.DataFrame:
    """
    A convenience function combining get_text_paths and readtext \
        to generate a polars DataFrame formatted for docuscope_parse.

    :param directory: A string representing the path \
        to a directory of text (TXT) files to be processed.
    :return: A polars DataFrame with 'doc_id' and 'text' columns.
    """
    text_files = get_text_paths(directory)
    if len(text_files) == 0:
        raise ValueError("""
                    No text files found in directory.
                    """)
    df = readtext(text_files)
    return df


def dtm_weight(dtm: pl.DataFrame,
               scheme="prop") -> pl.DataFrame:
    """
    A function for weighting a document-term-matrix.

    :param dtm: A document-term-matrix with a 'doc_id' column.
    :param scheme: One of 'prop' (normalized by totals per document), \
        'scale' (z-scores), \
            or 'tfidf' (term-frequency-inverse-document-frequency).
    :return: A polars DataFrame of weighted values.
    """
    if dtm.columns[0] != "doc_id":
        raise ValueError("""
                        Invalid DataFrame.
                        Expected a DataFrame produced by tags_dtm with 'doc_id' as the first column.
                        """)  # noqa: E501
    if not all(pl.UInt32 for x in dtm.collect_schema().dtypes()[1:]):
        raise ValueError("""
                        Invalid DataFrame.
                        All columns except 'doc_id' must be numeric.
                        """)

    scheme_types = ['prop', 'scale', 'tfidf']
    if scheme not in scheme_types:
        raise ValueError("""scheme_types
                         Invalid count_by type. Expected one of: %s
                         """ % scheme_types)

    weighted_df = (
        dtm
        .with_columns(
            pl.selectors.numeric()
            .truediv(
                pl.sum_horizontal(
                    pl.selectors.numeric()
                )
            )
        )
    )

    if scheme == "prop":
        return weighted_df

    elif scheme == "scale":
        weighted_df = (
            weighted_df
            .with_columns(
                pl.selectors.numeric()
                .sub(
                    pl.selectors.numeric().mean()
                    )
                .truediv(
                    pl.selectors.numeric().std()
                    )
                )
        )
        return weighted_df

    else:
        weighted_df = (
            weighted_df
            .transpose(include_header=True,
                       header_name="Tag",
                       column_names="doc_id")
            # log(1 + N/(1+df)) = log((1+df+N)/(1+df)) =
            # log(1+df+N) - log(1+df) = log1p(df+N) - log1p(df)
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().ge(0))
                .add(pl.sum_horizontal(pl.selectors.numeric().gt(0))).log1p()
                .sub(pl.sum_horizontal(pl.selectors.numeric().gt(0)).log1p())
                .alias("IDF")
            )
            # multiply normalized frequencies by IDF
            .with_columns(
                pl.selectors.numeric().exclude("IDF").mul(pl.col("IDF"))
            )
            .drop("IDF")
            .transpose(include_header=True,
                       header_name="doc_id",
                       column_names="Tag")
            )
        return weighted_df


def dtm_simplify(dtm: pl.DataFrame) -> pl.DataFrame:
    """
    A function for aggregating part-of-speech tags \
        into more general lexical categories \
            returning the equivalent of the tags_dtm function.

    :param dtm: A document-term-matrix with a doc_id column.
    :return: A polars DataFrame of absolute frequencies, \
        normalized frequencies(per million tokens) and ranges.
    """
    if dtm.columns[0] != "doc_id":
        raise ValueError("""
                        Invalid DataFrame.
                        Expected a DataFrame produced by tags_dtm with 'doc_id' as the first column.
                        """)  # noqa: E501
    if not all(pl.UInt32 for x in dtm.collect_schema().dtypes()[1:]):
        raise ValueError("""
                        Invalid DataFrame.
                        All columns except 'doc_id' must be numeric.
                        """)
    tag_prefix = ["NN", "VV", "II"]
    if (not any(
        x.startswith(tuple(tag_prefix)) for x in
        dtm.columns[1:]
                )):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a dtm with part-of-speech tags.
                         """)

    simple_dtm = (
        dtm
        .unpivot(pl.selectors.numeric(), index="doc_id")
        .with_columns(
            pl.col("variable")
            .str.replace('^NN\\S*$', '#NounCommon')
            .str.replace('^VV\\S*$', '#VerbLex')
            .str.replace('^J\\S*$', '#Adjective')
            .str.replace('^R\\S*$', '#Adverb')
            .str.replace('^P\\S*$', '#Pronoun')
            .str.replace('^I\\S*$', '#Preposition')
            .str.replace('^C\\S*$', '#Conjunction')
            .str.replace('^N\\S*$', '#NounOther')
            .str.replace('^VB\\S*$', '#VerbBe')
            .str.replace('^V\\S*$', '#VerbOther')
            )
        .with_columns(
             pl.when(pl.col("variable").str.starts_with("#"))
             .then(pl.col("variable"))
             .otherwise(pl.col("variable").str.replace('^\\S+$', '#Other'))
             )
        .with_columns(
            pl.col("variable").str.replace("#", "")
            )
        .group_by(["doc_id", "variable"], maintain_order=True).sum()
        .pivot(index="doc_id", on="variable", values="value")
        )

    return simple_dtm


def freq_simplify(frequency_table: pl.DataFrame) -> pl.DataFrame:
    """
    A function for aggregating part-of-speech tags \
        into more general lexical categories \
            returning the equivalent of the frequency_table function.

    :param frequency_table: A frequency table.
    :return: A polars DataFrame of token counts.
    """
    required_columns = {'Token', 'Tag', 'AF', 'RF', 'Range'}
    table_columns = set(frequency_table.columns)
    if not required_columns.issubset(table_columns):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by frequency_table
                         that includes columns: Token, Tag, AF, RF, Range.
                         """)
    tag_prefix = ["NN", "VV", "II"]
    if (not any(
        x.startswith(tuple(tag_prefix)) for x in
        frequency_table.get_column("Tag").to_list()
                )):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a frequency table with part-of-speech tags.
                         """)

    simple_df = (
        frequency_table
        .with_columns(
            pl.selectors.starts_with("Tag")
            .str.replace('^NN\\S*$', '#NounCommon')
            .str.replace('^VV\\S*$', '#VerbLex')
            .str.replace('^J\\S*$', '#Adjective')
            .str.replace('^R\\S*$', '#Adverb')
            .str.replace('^P\\S*$', '#Pronoun')
            .str.replace('^I\\S*$', '#Preposition')
            .str.replace('^C\\S*$', '#Conjunction')
            .str.replace('^N\\S*$', '#NounOther')
            .str.replace('^VB\\S*$', '#VerbBe')
            .str.replace('^V\\S*$', '#VerbOther')
        )
        .with_columns(
            pl.when(pl.selectors.starts_with("Tag").str.starts_with("#"))
            .then(pl.selectors.starts_with("Tag"))
            .otherwise(
                pl.selectors.starts_with("Tag").str.replace('^\\S+$', '#Other')
                ))
        .with_columns(
            pl.selectors.starts_with("Tag").str.replace("#", "")
        ))

    return simple_df


def tags_simplify(dtm: pl.DataFrame) -> pl.DataFrame:
    """
    A function for aggregating part-of-speech tags \
        into more general lexical categories \
            returning the equivalent of the tags_table function.

    :param dtm: A document-term-matrix with a doc_id column
    :return: A polars DataFrame of absolute frequencies, \
        normalized frequencies(per million tokens) and ranges.
    """
    if dtm.columns[0] != "doc_id":
        raise ValueError("""
                        Invalid DataFrame.
                        Expected a DataFrame produced by tags_dtm with 'doc_id' as the first column.
                        """)  # noqa: E501
    if not all(pl.UInt32 for x in dtm.collect_schema().dtypes()[1:]):
        raise ValueError("""
                        Invalid DataFrame.
                        All columns except 'doc_id' must be numeric.
                        """)
    tag_prefix = ["NN", "VV", "II"]
    if (not any(
        x.startswith(tuple(tag_prefix)) for x in
        dtm.columns[1:]
                )):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a dtm with part-of-speech tags.
                         """)

    dtm = dtm_simplify(dtm)
    simple_df = (
        dtm
        .transpose(
            include_header=True, header_name="Tag", column_names="doc_id")
        .with_columns(
                pl.sum_horizontal(pl.selectors.numeric() > 0)
                .alias("Range")
        )
        .with_columns(
            pl.col("Range").truediv(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("Range").is_not_null())
                ).mul(100)
        )
        # calculate absolute frequency
        .with_columns(
            pl.sum_horizontal(pl.selectors.numeric().exclude("Range"))
            .alias("AF")
        )
        .sort("AF", descending=True)
        .select(["Tag", "AF", "Range"])
        # calculate relative frequency
        .with_columns(
            pl.col("AF").truediv(pl.sum("AF")).mul(100)
            .alias("RF")
        )
        .select(["Tag", "AF", "RF", "Range"])
        )

    return simple_df


def dtm_to_coo(dtm: pl.DataFrame) -> coo_matrix:
    """
    A function for converting a tags dtm to a COOrdinate format.

    :param dtm: A document-term-matrix with a doc_id column
    :return: A COOrdinate format matrix, \
        an index of document ids, \
            and a list of variable names.
    """
    docs = dtm["doc_id"].to_list()
    vocab = dtm.drop("doc_id").columns
    matrix_values = dtm.drop("doc_id").to_numpy()
    coo_sparse_matrix = coo_matrix(matrix_values)
    return coo_sparse_matrix, docs, vocab


def from_tmtoolkit(tmtoolkit_corpus) -> pl.DataFrame:
    """
    A simple wrapper for coverting a tmtoolkit corpus \
        to a polars DataFrame.

    :param tmtoolkit_corpus: A tmtoolkit corpus.
    :return: A polars DataFrame formatted for \
        further process in docuscospacy.
    """
    if tmtoolkit_corpus.language_model != 'en_docusco_spacy':
        raise ValueError("""
                         Invalid spaCy model. Expected 'en_docusco_spacy'.
                         For information and instructions see:
                         https://huggingface.co/browndw/en_docusco_spacy
                         """)
    required_attributes = ['tag', 'ent_type', 'ent_iob']
    if (
        not all(
            item in list(tmtoolkit_corpus.spacy_token_attrs
                         ) for item in required_attributes)
    ):
        raise ValueError("""
                         Missing spaCy attributes. Expected all of: %s
                         """ % required_attributes)

    df_list = []
    for i in range(len(tmtoolkit_corpus.keys())):
        doc_id = list(tmtoolkit_corpus.keys())[i]
        token_list = tmtoolkit_corpus[i]["token"]
        ws_list = tmtoolkit_corpus[i]["whitespace"]
        tag_list = tmtoolkit_corpus[i]["tag"]
        iob_list = tmtoolkit_corpus[i]["ent_iob"]
        iob_list = [x.replace('IS_DIGIT', 'B') for x in iob_list]
        iob_list = [x.replace('IS_ALPHA', 'I') for x in iob_list]
        iob_list = [x.replace('IS_ASCII', 'O') for x in iob_list]
        ent_list = tmtoolkit_corpus[i]["ent_type"]
        iob_ent = list(map('-'.join, zip(iob_list, ent_list)))
        df = pl.DataFrame({
            "doc_id": doc_id,
            "token": token_list,
            "ws": ws_list,
            "pos_tag": tag_list,
            "ds_tag": iob_ent
        })
        df_list.append(df)
    # contatenate list of DataFrames
    df_list.sort(key=lambda d: d['doc_id'][0])
    df = pl.concat(df_list)
    df = (
        df
        # assign unique ids to part-of-speech tags for grouping
        .with_columns(
            pl.when(
                pl.col("pos_tag").str.contains("\\d\\d$")
                & pl.col("pos_tag").str.contains("[^1]$")
            )
            .then(0)
            .otherwise(1)
            .cast(pl.UInt32, strict=False)
            .alias('pos_id')
        )
        .with_columns(
            pl.when(
                pl.col("pos_id") == 1)
            .then(pl.cum_sum("pos_id"))
            .otherwise(None)
            .forward_fill()
        )
        # ensure that ids and base tags are the same
        # (e.g., II21, II22, etc. render as II)
        .with_columns(
            pl.when(
                pl.col("pos_tag").str.contains("\\d\\d$") &
                pl.col("pos_tag").str.contains("[^1]$")
            )
            .then(None)
            .otherwise(pl.col("pos_tag").str.replace("\\d\\d$", ""))
            .forward_fill()
            .name.keep()
        )
        # assign unique ids to DocuScope tags for grouping
        .with_columns(
            pl.when(
                pl.col("ds_tag").str.starts_with("B-") |
                pl.col("ds_tag").str.starts_with("O-")
            )
            .then(1)
            .otherwise(0)
            .cast(pl.UInt32, strict=False)
            .alias('ds_id')
        )
        .with_columns(
            pl.when(
                pl.col("ds_id") == 1
            )
            .then(pl.cum_sum("ds_id"))
            .otherwise(None)
            .forward_fill()
        )
        # ensure that ids and base tags are the same (e.g., B-ConfidenceHigh,
        # I-ConfidenceHigh are rendered as ConfidenceHigh)
        .with_columns(
            pl.when(
                pl.col("ds_tag").str.starts_with("B-") |
                pl.col("ds_tag").str.starts_with("O-")
            )
            .then(pl.col("ds_tag").str.strip_chars_start("B-"))
            .otherwise(None)
            .forward_fill()
        )
        .with_columns(
            pl.when(
                pl.col("ds_tag") == "O-"
            )
            .then(pl.col("ds_tag").str.replace("O-", "Untagged"))
            .otherwise(pl.col("ds_tag"))
        )
        .with_columns(
            pl.when(
                pl.col("token").str.contains("^[[:punct:]]+$")
            )
            .then(pl.lit("Y").alias("pos_tag"))
            .otherwise(pl.col("pos_tag"))
        )
        .with_columns(
            pl.col("token")
            .shift(1)
            .alias("token_1")
        )
        .with_columns(
            pl.when(
                (
                    pl.col("token").str.contains(r"(?i)^s$")
                    ) &
                (
                    pl.col("token_1").str.contains(r"(?i)^it$")
                )
            )
            .then(pl.lit("GE").alias("pos_tag"))
            .otherwise(pl.col("pos_tag"))
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("token"),
                    pl.col("ws")
                ],
                separator="",
            ).alias("token")
        )
        .drop(["token_1", "ws"])
    )
    return df


def convert_corpus(*args):
    warnings.warn("convert_corpus is deprecated, use from_tmtoolkit instead.",
                  DeprecationWarning, stacklevel=2)
    # ... old implementation ...
