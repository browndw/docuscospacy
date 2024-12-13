"""
Functions for analyzing corpus data tagged with DocuScope and CLAWS7.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import re
import math
import unicodedata

from collections import OrderedDict
from typing import Union, List

import polars as pl
import numpy as np
from spacy.tokens import Doc
from spacy.language import Language
from scipy.stats.distributions import chi2


def _str_squish(text: str) -> str:
    """
    Remove extra spaces, returns, etc. from a string.

    :param text: A string.
    :return: A string.
    """
    return " ".join(text.split())


def _replace_curly_quotes(text: str) -> str:
    """
    Replaces curly quotes with straight quotes.

    :param text: A string.
    :return: A string.
    """
    text = text.replace(u'\u2018', "'")  # Left single quote
    text = text.replace(u'\u2019', "'")  # Right single quote
    text = text.replace(u'\u201C', '"')  # Left double quote
    text = text.replace(u'\u201D', '"')  # Right double quote
    return text


def _split_docs(doc_txt: str,
                n_chunks: float) -> List:
    """
    Splits documents that will exhaust spaCy's memory into smaller chunks.

    :param doc_txt: A string.
    :param n_chunks: The number of chunks for splitting.
    :return: A list of strings.
    """
    sent_boundary = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    doc_len = len(doc_txt)
    chunk_idx = [math.ceil(i/n_chunks*doc_len) for i in range(1, n_chunks)]
    split_idx = [sent_boundary.search(
        doc_txt[idx:]
        ).span()[1] + (idx-1) for idx in chunk_idx]
    split_idx.insert(0, 0)
    doc_chunks = [doc_txt[i:j] for i, j in zip(
        split_idx, split_idx[1:] + [None]
        )]
    if len(doc_chunks) == n_chunks:
        return doc_chunks
    else:
        split_idx = [re.search(
            ' ', doc_txt[idx:]
            ).span()[0] + idx for idx in chunk_idx]
        split_idx.insert(0, 0)
        doc_chunks = [doc_txt[i:j] for i, j in zip(
            split_idx, split_idx[1:] + [None]
            )]
        return doc_chunks


def _pre_process_corpus(corp: pl.DataFrame) -> pl.DataFrame:
    """
    Format texts to increase spaCy tagging accuracy.

    :param corp: A polars DataFrame with 'doc_id' and 'text' columns.
    :return: A polars DataFrame.
    """
    df = (
        corp
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: _str_squish(x),
                          return_dtype=pl.String)
                        )
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: _replace_curly_quotes(x),
                          return_dtype=pl.String)
                        )
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: unicodedata.normalize('NFKD', x)
                          .encode('ascii', errors='ignore')
                          .decode('utf-8'), return_dtype=pl.String)
                        )
        .with_columns(
            pl.col("text")
            .str.replace_all(r"(?i)\b(it)(s)\b", "${1} ${2}")
            )
    )
    return df


def _calc_disp2(v: pl.Series,
                total: float,
                s=None) -> pl.DataFrame:
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
    values["Lynes_D3"] = 1 - (
        np.sum(((v - np.mean(v)) ** 2) / np.mean(v)) / (4 * f)
        )
    values["DC"] = ((np.sum(np.sqrt(v)) / n) ** 2) / np.mean(v)
    values["Juillands_D"] = 1 - (
        np.std(v / s) / np.mean(v / s)
        ) / np.sqrt(len(v / s) - 1)

    values["DP"] = np.sum(np.abs((v / f) - s)) / 2
    # corrected
    values["DP_norm"] = (np.sum(np.abs((v / f) - s)) / 2) / (1 - np.min(s))

    return pl.DataFrame(values)


def docuscope_parse(corp: pl.DataFrame,
                    nlp_model: Language,
                    n_process=1,
                    batch_size=25) -> pl.DataFrame:
    """
    Parse a corpus using the 'en_docuso_spacy' model.

    :param corp: A polars DataFrame \
        conataining a 'doc_id' column and a 'text' column.
    :param nlp_model: An 'en_docuso_spacy' instance.
    :param n_process: The number of parallel processes \
        to use during parsing.
    :param n_process: The batch size to use during parsing.
    :return: a polars DataFrame with, \
        token sequencies identified by both part-of-speech tags \
        and DocuScope tags.
    """
    validation = OrderedDict([('doc_id', pl.String),
                              ('text', pl.String)])
    if corp.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame with 2 columns (doc_id & text).
                         """)
    if nlp_model.lang + '_' + nlp_model.meta['name'] != 'en_docusco_spacy':
        raise ValueError("""
                         Invalid spaCy model. Expected 'en_docusco_spacy'.
                         For information and instructions see:
                         https://huggingface.co/browndw/en_docusco_spacy
                         """)

    corp = _pre_process_corpus(corp)
    # split long texts (> 500000 chars) into chunks
    corp = (corp
            .with_columns(
                n_chunks=pl.Expr.ceil(
                    pl.col("text").str.len_chars().truediv(500000)
                )
                .cast(pl.UInt32, strict=False)
                )
            .with_columns(
                chunk_id=pl.int_ranges("n_chunks")
                )
            .with_columns(
                pl.struct(['text', 'n_chunks'])
                .map_elements(lambda x: _split_docs(x['text'], x['n_chunks']),
                              return_dtype=pl.List(pl.String))
                .alias("text")
                )
            .explode("text", "chunk_id")
            .with_columns(
                pl.col("text").str.strip_chars() + " "
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("chunk_id"),
                        pl.col("doc_id")
                        ], separator="@",
                        ).alias("doc_id")
                    )
            .drop(["n_chunks", "chunk_id"])
            )
    # tuple format for spaCy
    text_tuples = []
    for item in corp.to_dicts():
        text_tuples.append((item['text'], {"doc_id": item['doc_id']}))
    # add doc_id as custom attribute
    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    # create pipeline
    doc_tuples = nlp_model.pipe(text_tuples,
                                as_tuples=True,
                                n_process=n_process,
                                batch_size=batch_size)
    # process corpus and gather into a DataFrame
    df_list = []
    for doc, context in doc_tuples:
        doc._.doc_id = context["doc_id"]
        token_list = [token.text for token in doc]
        ws_list = [token.whitespace_ for token in doc]
        tag_list = [token.tag_ for token in doc]
        iob_list = [token.ent_iob_ for token in doc]
        ent_list = [token.ent_type_ for token in doc]
        iob_ent = list(map('-'.join, zip(iob_list, ent_list)))
        df = pl.DataFrame({
            "doc_id": doc._.doc_id,
            "token": token_list,
            "ws": ws_list,
            "pos_tag": tag_list,
            "ds_tag": iob_ent
        })
        df_list.append(df)
    # contatenate list of DataFrames
    df = pl.concat(df_list)
    # add tag ids and format
    df = (
        df
        .with_columns(
            pl.col("doc_id")
            .str.split_exact("@", 1)
            )
        .unnest("doc_id")
        .rename({"field_0": "chunk_id", "field_1": "doc_id"})
        .with_columns(
            pl.col("chunk_id")
            .cast(pl.UInt32, strict=False)
            )
        .sort(["doc_id", "chunk_id"], descending=[False, False])
        .drop("chunk_id")
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


def frequency_table(tokens_table: pl.DataFrame,
                    count_by="pos") -> pl.DataFrame:
    """
    Generate a count of token frequencies.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds', or 'both' for aggregating tokens
    :return: A polars DataFrame of token counts
    """
    count_types = ['pos', 'ds', 'both']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    def summarize_counts(df):
        df = (
            df
            .pivot(
                index="Token", on="doc_id",
                values="len", aggregate_function="sum"
                )
            .with_columns(
                pl.all().exclude("Token").cast(pl.UInt32, strict=True)
            )
            # calculate range
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().is_not_null())
                .alias("Range")
            )
            .with_columns(
                pl.selectors.numeric().fill_null(strategy="zero")
            )
            # normalize over total documents in corpus
            .with_columns(
                    pl.col("Range").truediv(
                        pl.sum_horizontal(
                            pl.selectors.numeric()
                            .exclude("Range")
                            .is_not_null()
                            )
                        ).mul(100)
            )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().exclude("Range"))
                .alias("AF")
            )
            .sort("AF", descending=True)
            .select(["Token", "AF", "Range"])
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.sum("AF")).mul(1000000)
                .alias("RF")
            )
            # format data
            .unnest("Token")
            .select(["Token", "Tag", "AF", "RF", "Range"])
            )
        return (df)

    # format tokens and sum by doc_id
    df_pos = (
        tokens_table
        .group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
        )
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .filter(
            pl.col("pos_tag") != "Y"
        )
        .rename({"pos_tag": "Tag"})
        .rename({"token": "Token"})
        .group_by(["doc_id", "Token", "Tag"]).len()
        .with_columns(
            pl.struct(["Token", "Tag"])
        )
        .select(pl.exclude("Tag"))
        )

    df_pos = summarize_counts(df_pos).sort(
        ["AF", "Token"], descending=[True, False]
        )

    if count_by == "pos":
        return (df_pos)
    else:
        df_ds = (
            tokens_table
            .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
            .agg(
                pl.col("token").str.concat("")
            )
            .with_columns(
                pl.col("token").str.to_lowercase().str.strip_chars())
            .filter(
                ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                  pl.col("ds_tag").str.contains("Untagged"))
            )
            .rename({"ds_tag": "Tag"})
            .rename({"token": "Token"})
            .group_by(["doc_id", "Token", "Tag"]).len()
            .with_columns(
                pl.struct(["Token", "Tag"])
            )
            .select(pl.exclude("Tag"))
            )

        df_ds = summarize_counts(df_ds).sort(["AF", "Token"],
                                             descending=[True, False])
    if count_by == "ds":
        return (df_ds)
    else:
        return (df_pos, df_ds)


def tags_table(tokens_table: pl.DataFrame,
               count_by='pos') -> pl.DataFrame:
    """
    Generate a count of tag frequencies.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
    :return: a polars DataFrame of absolute frequencies, \
        normalized frequencies(per million tokens) and ranges
    """
    count_types = ['pos', 'ds', 'both']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    def summarize_counts(df):
        df = (
            df
            .pivot(index="Tag",
                   on="doc_id",
                   values="len",
                   aggregate_function="sum")
            .with_columns(
                pl.all().exclude("Tag").cast(pl.UInt32, strict=True)
            )
            # calculate range
            .with_columns(
                Range=pl.sum_horizontal(pl.selectors.numeric().is_not_null())
            )
            .with_columns(
                pl.selectors.numeric().fill_null(strategy="zero")
            )
            # normalize over total documents in corpus
            .with_columns(
                    pl.col("Range").truediv(
                        pl.sum_horizontal(
                            pl.selectors.numeric()
                            .exclude("Range").is_not_null()
                            )
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
        return (df)

    # format tokens and sum by doc_id
    df_pos = (
        tokens_table
        .filter(pl.col("pos_tag") != "Y")
        .group_by(["doc_id", "pos_id", "pos_tag"], maintain_order=True)
        .first()
        .group_by(["doc_id", "pos_tag"]).len()
        .rename({"pos_tag": "Tag"})
        )

    df_pos = summarize_counts(df_pos).sort(["AF", "Tag"],
                                           descending=[True, False])

    if count_by == "pos":
        return df_pos
    else:
        df_ds = (
            tokens_table
            .filter(~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                      pl.col("ds_tag").str.contains("Untagged")))
            .group_by(["doc_id", "ds_id", "ds_tag"], maintain_order=True)
            .first()
            .group_by(["doc_id", "ds_tag"]).len()
            .rename({"ds_tag": "Tag"})
            )

        df_ds = summarize_counts(df_ds).sort(["AF", "Tag"],
                                             descending=[True, False])
    if count_by == "ds":
        return df_ds
    else:
        return df_pos, df_ds


def dispersions_table(tokens_table: pl.DataFrame,
                      count_by='pos') -> pl.DataFrame:
    """
    Generate a table of dispersion measures.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a polars DataFrame with various dispersion measures.
    """
    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    if count_by == "pos":
        dtm = (
            tokens_table
            .with_columns(
                pl.col("token").str.to_lowercase().str.strip_chars()
                )
            .filter(
                pl.col("pos_tag") != "Y"
                )
            .group_by(
                ["doc_id", "pos_id", "pos_tag"], maintain_order=True
                )
            .first()
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token"),
                        pl.col("pos_tag")
                        ], separator="_",
                        ).alias("pos_tag"),
                        )
            .group_by(
                ["doc_id", "pos_tag"]
                )
            .len()
            .with_columns(
                pl.col("len").sum().over('pos_tag').alias('total')
                )
            .sort(
                ["total", "doc_id"], descending=[True, False]
                )
            .pivot(
                index="doc_id",
                on="pos_tag",
                values="len",
                aggregate_function="sum"
                )
            .fill_null(
                strategy="zero"
                )
            )
    if count_by == "ds":
        dtm = (
            tokens_table
            .with_columns(
                pl.col("token").str.to_lowercase().str.strip_chars()
                )
            .filter(
                ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                  pl.col("ds_tag").str.contains("Untagged"))
                  )
            .group_by(
                ["doc_id", "pos_id", "ds_tag"], maintain_order=True
                )
            .first()
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token"),
                        pl.col("ds_tag")
                        ], separator="_",
                        ).alias("ds_tag"),
                        )
            .group_by(
                ["doc_id", "ds_tag"]
                ).len()
            .with_columns(
                pl.col("len").sum().over('ds_tag').alias('total')
                )
            .sort(
                ["total", "doc_id"], descending=[True, False]
                )
            .pivot(
                index="doc_id",
                on="ds_tag",
                values="len",
                aggregate_function="sum"
                )
            .fill_null(
                strategy="zero"
                )
            )

    total = dtm.drop("doc_id").sum().sum_horizontal().item()
    parts = (dtm.drop("doc_id").sum_horizontal() / total).to_numpy()
    idx = range(1, dtm.width)
    dsp = [_calc_disp2(dtm[:, i], total, parts) for i in idx]
    dsp = pl.concat(dsp)
    return dsp


def tags_dtm(tokens_table: pl.DataFrame,
             count_by='pos') -> pl.DataFrame:
    """
    Generate a document-term matrix of raw tag counts.

    :param tokens_table: A polars DataFrame \
        as generated by the docuscope_parse function
    :param count_by: One of 'pos', 'ds' or 'both' for aggregating tokens
    :return: a polars DataFrame of absolute tag frequencies for each document
    """
    count_types = ['pos', 'ds', 'both']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    df_pos = (
        tokens_table
        .filter(
            pl.col("pos_tag") != "Y"
            )
        .group_by(
            ["doc_id", "pos_id", "pos_tag"], maintain_order=True
            )
        .first()
        .group_by(
            ["doc_id", "pos_tag"]
            ).len()
        .rename(
            {"pos_tag": "tag"}
            )
        .with_columns(
            pl.col("len").sum().over('tag').alias('total')
            )
        .sort(
            ["total", "doc_id"], descending=[True, False]
            )
        .pivot(
            index="doc_id", on="tag", values="len", aggregate_function="sum"
            )
        .fill_null(
            strategy="zero"
            )
        )

    if count_by == "pos":
        return df_pos

    df_ds = (
        tokens_table
        .filter(
            ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
              pl.col("ds_tag").str.contains("Untagged"))
              )
        .group_by(
            ["doc_id", "ds_id", "ds_tag"], maintain_order=True
            )
        .first()
        .group_by(
            ["doc_id", "ds_tag"]
            ).len()
        .rename(
            {"ds_tag": "tag"}
            )
        .with_columns(
            pl.col("len").sum().over('tag').alias('total')
            )
        .sort(
            ["total", "doc_id"], descending=[True, False]
            )
        .pivot(
            index="doc_id", on="tag", values="len", aggregate_function="sum"
            )
        .fill_null
        (strategy="zero"
         )
    )

    if count_by == "ds":
        return df_ds
    else:
        return df_pos, df_ds


def ngrams(tokens_table: pl.DataFrame,
           span=2,
           min_frequency=10,
           count_by='pos') -> pl.DataFrame:
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
    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)
    if span < 2 or span > 5:
        raise ValueError("Span must be < " + str(2) + " and > " + str(5))

    if count_by == 'pos':
        grouping_tag = "pos_tag"
        grouping_id = "pos_id"
        expr_filter = pl.col("pos_tag") != "Y"
        struct_labels = ["token_" + str(i) for i in range(span)]
    else:
        grouping_tag = "ds_tag"
        grouping_id = "ds_id"
        expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                        pl.col("ds_tag").str.contains("Untagged"))
        struct_labels = ["token_" + str(i) for i in range(span)]

    look_around_token = [
        pl.col("token")
        .shift(-i).alias(f"tok_lag_{i}") for i in range(span)
        ]
    look_around_tag = [
        pl.col(grouping_tag)
        .shift(-i).alias(f"tag_lag_{i}") for i in range(span)
        ]

    rename_tokens = [
            pl.col('ngram').struct.rename_fields(
                [f'Token_{i + 1}' for i in range(span)]
                )]
    rename_tags = [
            pl.col('tags').struct.rename_fields(
                [f'Tag_{i + 1}' for i in range(span)]
                )]

    ngram_df = (
        tokens_table
        .group_by(["doc_id", grouping_id, grouping_tag], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .filter(expr_filter)
        .with_columns(pl.col("token").len().alias("total"))
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .with_columns(
            look_around_token + look_around_tag
            )
        .group_by("doc_id", "total")
        .agg(
            pl.concat_list([f"tok_lag_{i}" for i in range(span)]
                           ).alias("ngram"),
            pl.concat_list([f"tag_lag_{i}" for i in range(span)]
                           ).alias("tags")
            )
        .explode(["ngram", "tags"])
        .with_columns(
            pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
            )
        .group_by(
            ["doc_id", "total", "ngram", "tags"]
            ).len().sort("len", descending=True)
        .with_columns(
            pl.struct(["ngram", "tags"])
            )
        .select(pl.exclude("tags"))
        .pivot(index=["ngram", "total"],
               on="doc_id", values="len",
               aggregate_function="sum")
        .with_columns(
            pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
            )
        # calculate range
        .with_columns(
            pl.sum_horizontal(
                pl.selectors.numeric().exclude("total").is_not_null()
                ).alias("Range")
            )
        .with_columns(
            pl.selectors.numeric().fill_null(strategy="zero")
            )
        # normalize over total documents in corpus
        .with_columns(
                pl.col("Range").truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric().exclude(
                            ["Range", "total"]).is_not_null()
                        )
                    ).mul(100)
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
            pl.col("AF").truediv(pl.col("total")).mul(1000000)
            .alias("RF")
            )
        .select(["ngram", "AF", "RF", "Range"])
        .unnest("ngram")
        .with_columns(
            rename_tokens + rename_tags
            )
        .unnest(["ngram", "tags"])
        .sort(["AF", "Token_1", "Token_2"], descending=[True, False, False])
        .filter(
                pl.col('RF') >= min_frequency
            )
        )

    return ngram_df


def clusters_by_token(tokens_table: pl.DataFrame,
                      node_word: str,
                      node_position=1,
                      span=2,
                      search_type="fixed",
                      count_by='pos'):
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
    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    if node_position > span:
        node_position = span
        print('Setting node position to right-most position in span.')
    if span < 2 or span > 5:
        raise ValueError("Span must be < " + str(2) + " and > " + str(5))
    if search_type not in ['fixed', 'starts_with', 'ends_with', 'contains']:
        raise ValueError("""
                         Search type must be on of 'fixed',
                         'starts_with', 'ends_with', or 'contains'.
                         """)

    if count_by == 'pos':
        grouping_tag = "pos_tag"
        grouping_id = "pos_id"
        expr_filter = pl.col("pos_tag") != "Y"
        struct_labels = ["token_" + str(i) for i in range(span)]
    else:
        grouping_tag = "ds_tag"
        grouping_id = "ds_id"
        expr_filter = ~(
            pl.col("token").str.contains("^[[[:punct:]] ]+$") &
            pl.col("ds_tag").str.contains("Untagged")
            )
        struct_labels = ["token_" + str(i) for i in range(span)]

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
        pl.col("token")
        .shift(-i).alias(f"tok_lag_{i}") for i in range(
            -preceding, following + 1
            )]
    look_around_tag = [
        pl.col(grouping_tag)
        .shift(-i).alias(f"tag_lag_{i}") for i in range(
            -preceding, following + 1
            )]

    rename_tokens = [
            pl.col('ngram')
            .struct.rename_fields([f'Token_{i + 1}' for i in range(span)])
            ]
    rename_tags = [
            pl.col('tags')
            .struct.rename_fields([f'Tag_{i + 1}' for i in range(span)])
            ]

    ngram_df = (
        tokens_table
        .group_by(["doc_id", grouping_id, grouping_tag], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .filter(expr_filter)
        .with_columns(pl.col("token").len().alias("total"))
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .with_columns(
            look_around_token + look_around_tag
            )
        .filter(expr)
        .group_by("doc_id", "total")
        .agg(
            pl.concat_list(
                [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("ngram"),
            pl.concat_list(
                [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("tags")
            )
        .explode(["ngram", "tags"])
    )

    if ngram_df.height == 0:
        return ngram_df

    else:
        ngram_df = (
            ngram_df
            .with_columns(
                pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
                )
            .group_by(["doc_id", "total", "ngram", "tags"])
            .len().sort("len", descending=True)
            .with_columns(
                pl.struct(["ngram", "tags"])
                    )
            .select(pl.exclude("tags"))
            .pivot(index=["ngram", "total"],
                   on="doc_id",
                   values="len",
                   aggregate_function="sum")
            .with_columns(
                pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
                )
            # calculate range
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("total").is_not_null()
                    )
                .alias("Range")
                )
            .with_columns(
                pl.selectors.numeric().fill_null(strategy="zero")
                )
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range").truediv(
                    pl.sum_horizontal(
                        pl.selectors.numeric()
                        .exclude(["Range", "total"]).is_not_null()
                        )
                    ).mul(100)
                )
            # calculate absolute frequency
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric()
                                  .exclude(["Range", "total"]))
                .alias("AF")
                )
            .sort("AF", descending=True)
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.col("total")).mul(1000000)
                .alias("RF")
                )
            .select(["ngram", "AF", "RF", "Range"])
            .unnest("ngram")
            .with_columns(
                rename_tokens + rename_tags
                )
            .unnest(["ngram", "tags"])
        )
        return ngram_df


def clusters_by_tag(tokens_table: pl.DataFrame,
                    tag: str,
                    tag_position=1,
                    span=2,
                    count_by='pos') -> pl.DataFrame:
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
    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)
    if tag_position > span:
        tag_position = span
        print('Setting node position to right-most position in span.')
    if span < 2 or span > 5:
        raise ValueError("Span must be < " + str(2) + " and > " + str(5))

    if count_by == 'pos':
        grouping_tag = "pos_tag"
        grouping_id = "pos_id"
        expr = pl.col("pos_tag") == tag
        expr_filter = pl.col("pos_tag") != "Y"
        struct_labels = ["token_" + str(i) for i in range(span)]
    else:
        grouping_tag = "ds_tag"
        grouping_id = "ds_id"
        expr = pl.col("ds_tag") == tag
        expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                        pl.col("ds_tag").str.contains("Untagged"))
        struct_labels = ["token_" + str(i) for i in range(span)]

    preceding = tag_position - 1
    following = span - tag_position

    look_around_token = [
        pl.col("token")
        .shift(-i).alias(f"tok_lag_{i}") for i in range(
            -preceding, following + 1
            )]
    look_around_tag = [
        pl.col(grouping_tag)
        .shift(-i).alias(f"tag_lag_{i}") for i in range(
            -preceding, following + 1
            )]

    rename_tokens = [
            pl.col('ngram').struct
            .rename_fields([f'Token_{i + 1}' for i in range(span)])
            ]
    rename_tags = [
            pl.col('tags').struct
            .rename_fields([f'Tag_{i + 1}' for i in range(span)])
            ]

    ngram_df = (
        tokens_table
        .group_by(["doc_id", grouping_id, grouping_tag], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .filter(expr_filter)
        .with_columns(pl.col("token").len().alias("total"))
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .with_columns(
            look_around_token + look_around_tag
            )
        .filter(expr)
        .group_by("doc_id", "total")
        .agg(
            pl.concat_list(
                [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("ngram"),
            pl.concat_list(
                [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("tags")
            )
        .explode(["ngram", "tags"])
    )

    if ngram_df.height == 0:
        return ngram_df

    else:
        ngram_df = (
            ngram_df
            .with_columns(
                pl.col(["ngram", "tags"]).list.to_struct(fields=struct_labels)
                )
            .group_by(
                ["doc_id", "total", "ngram", "tags"]
                ).len().sort("len", descending=True)
            .with_columns(
                pl.struct(["ngram", "tags"])
                )
            .select(pl.exclude("tags"))
            .pivot(index=["ngram", "total"],
                   on="doc_id", values="len",
                   aggregate_function="sum")
            .with_columns(
                pl.all().exclude("ngram").cast(pl.UInt32, strict=True)
                )
            # calculate range
            .with_columns(
                pl.sum_horizontal(
                    pl.selectors.numeric().exclude("total").is_not_null()
                    )
                .alias("Range")
                )
            .with_columns(
                pl.selectors.numeric().fill_null(strategy="zero")
                )
            # normalize over total documents in corpus
            .with_columns(
                pl.col("Range").truediv(
                    pl.sum_horizontal(pl.selectors.numeric()
                                      .exclude(
                                          ["Range", "total"]
                                          ).is_not_null())
                    ).mul(100)
                )
            # calculate absolute frequency
            .with_columns(
                AF=pl.sum_horizontal(
                    pl.selectors.numeric().exclude(["Range", "total"])
                    )
                )
            .sort("AF", descending=True)
            # calculate relative frequency
            .with_columns(
                pl.col("AF").truediv(pl.col("total")).mul(1000000)
                .alias("RF")
                )
            .select(["ngram", "AF", "RF", "Range"])
            .unnest("ngram")
            .with_columns(
                rename_tokens + rename_tags
                )
            .unnest(["ngram", "tags"])
            )

        return ngram_df


def kwic_center_node(tokens_table: pl.DataFrame,
                     node_word: str,
                     ignore_case=True,
                     search_type='fixed') -> pl.DataFrame:
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
    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)

    if search_type not in ['fixed', 'starts_with', 'ends_with', 'contains']:
        raise ValueError("""
                         Search type must be on of 'fixed',
                         'starts_with', 'ends_with', or 'contains'.
                         """)

    if search_type == "fixed" and ignore_case:
        expr = pl.col("token"
                      ).str.to_lowercase(
                      ).str.strip_chars() == node_word.lower()
    if search_type == "fixed" and not ignore_case:
        expr = pl.col("token"
                      ).str.strip_chars() == node_word
    elif search_type == "starts_with" and ignore_case:
        expr = pl.col("token"
                      ).str.to_lowercase(
                      ).str.strip_chars().str.starts_with(node_word.lower())
    elif search_type == "starts_with" and not ignore_case:
        expr = pl.col("token"
                      ).str.strip_chars().str.starts_with(node_word)
    elif search_type == "ends_with" and ignore_case:
        expr = pl.col("token"
                      ).str.to_lowercase(
                      ).str.strip_chars().str.ends_with(node_word.lower())
    elif search_type == "ends_with" and not ignore_case:
        expr = pl.col("token"
                      ).str.ends_with(node_word)
    elif search_type == "contains" and ignore_case:
        expr = pl.col("token"
                      ).str.to_lowercase(
                      ).str.strip_chars().str.contains(node_word.lower())
    elif search_type == "contains" and not ignore_case:
        expr = pl.col("token"
                      ).str.strip_chars().str.contains(node_word)

    preceding = 7
    following = 7

    look_around_token = [
        pl.col("token").shift(-i).alias(
            f"tok_lag_{i}") for i in range(
                -preceding, following + 1)]

    kwic_df = (
        tokens_table
        .group_by(["doc_id", "pos_id"], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .with_columns(
            look_around_token
            )
        .filter(expr)
        .group_by("doc_id")
        .agg(
            pl.concat_list([f"tok_lag_{i}" for i in range(
                -preceding, following + 1)]).alias("node")
            )
        .explode("node")
        .with_columns(
            pre_node=pl.col("node").list.head(7)
        )
        .with_columns(
            post_node=pl.col("node").list.tail(7)
        )
        .with_columns(
            pl.col("node").list.get(7)
        )
        .with_columns(
            pl.col("pre_node").list.join("")
        )
        .with_columns(
            pl.col("post_node").list.join("")
        )
        .select(["doc_id", "pre_node", "node", "post_node"])
        .sort("doc_id")
        .rename({"doc_id": "Doc ID",
                 "pre_node": "Pre-Node",
                 "node": "Node",
                 "post_node": "Post-Node"})
    )

    return kwic_df


def coll_table(tokens_table: pl.DataFrame,
               node_word: str,
               preceding=4,
               following=4,
               statistic='npmi',
               count_by='pos',
               node_tag=None) -> pl.DataFrame:
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
    stat_types = ['pmi', 'npmi', 'pmi2', 'pmi3']
    if statistic not in stat_types:
        raise ValueError("""
                         Invalid statistic type. Expected one of: %s
                         """ % stat_types)

    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)
    if count_by == 'pos':
        grouping_tag = "pos_tag"
        grouping_id = "pos_id"
        expr_filter = pl.col("pos_tag") != "Y"
    else:
        grouping_tag = "ds_tag"
        grouping_id = "ds_id"
        expr_filter = ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") &
                        pl.col("ds_tag").str.contains("Untagged"))

    if node_tag is None:
        expr = pl.col("token") == node_word.lower()
    else:
        expr = (
            pl.col("token") == node_word.lower()
            ) & (pl.col(grouping_tag).str.starts_with(node_tag))

    look_around_token = [
        pl.col("token")
        .shift(-i).alias(f"tok_lag_{i}") for i in range(
            -preceding, following + 1
            )]

    look_around_tag = [
        pl.col(grouping_tag)
        .shift(-i).alias(f"tag_lag_{i}") for i in range(
            -preceding, following + 1
            )]

    total_df = (
        tokens_table
        .group_by(["doc_id", grouping_id, grouping_tag], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .filter(expr_filter)
        .group_by(["token", grouping_tag]).len(name="Freq_Total")
        .rename({"token": "Token", grouping_tag: "Tag"})
    )

    token_total = sum(total_df.get_column("Freq_Total"))

    if node_tag is None:
        node_freq = total_df.filter(
            pl.col("Token") == node_word
            ).get_column("Freq_Total").sum()

    else:
        node_freq = total_df.filter(
            (pl.col("Token") == node_word.lower()) &
            (pl.col("Tag").str.starts_with(node_tag))
            ).get_column("Freq_Total").sum()

    if node_freq == 0:
        coll_df = pl.DataFrame(schema=[("Token", pl.String),
                                       ("Tag", pl.String),
                                       ("Freq Span", pl.UInt32),
                                       ("Freq Total", pl.UInt32),
                                       ("MI", pl.Float64)])
        return coll_df

    if statistic == 'pmi':
        mi_funct = pl.col(
            "Freq_Span"
            ).truediv(token_total
                      ).log(base=2).sub(
                          pl.col(
                              "Freq_Total"
                              ).truediv(token_total
                                        ).mul(node_freq
                                              ).truediv(token_total
                                                        ).log(base=2)
                        )

    if statistic == 'npmi':
        mi_funct = pl.col(
            "Freq_Span"
            ).truediv(token_total
                      ).log(base=2).sub(
                          pl.col("Freq_Total"
                                 ).truediv(token_total
                                           ).mul(node_freq
                                                 ).truediv(token_total
                                                           ).log(base=2)
                        ).truediv(
                            pl.col("Freq_Span").truediv(token_total
                                                        ).log(base=2).neg()
                            )

    if statistic == 'pmi2':
        mi_funct = pl.col(
            "Freq_Span"
            ).truediv(token_total
                      ).log(base=2).sub(
                          pl.col("Freq_Total"
                                 ).truediv(token_total
                                           ).mul(node_freq
                                                 ).truediv(token_total
                                                           ).log(base=2)
                        ).sub(
                            pl.col("Freq_Span").truediv(token_total
                                                        ).log(base=2).mul(-1)
                            )

    if statistic == 'pmi3':
        mi_funct = pl.col(
            "Freq_Span"
            ).truediv(token_total
                      ).log(base=2).sub(
                pl.col("Freq_Total"
                       ).truediv(token_total
                                 ).mul(node_freq
                                       ).truediv(token_total
                                                 ).log(base=2)
                       ).sub(
                           pl.col("Freq_Span").truediv(token_total
                                                       ).log(base=2).mul(-2)
                           )

    coll_df = (
        tokens_table
        .group_by(["doc_id", grouping_id, grouping_tag], maintain_order=True)
        .agg(
            pl.col("token").str.concat("")
            )
        .with_columns(
            pl.col("token").str.to_lowercase().str.strip_chars())
        .filter(
            pl.col('token').str.contains("[a-z]")
            )
        .with_columns(
            look_around_token + look_around_tag
            )
        .filter(expr)
        # .drop(["tok_lag_0", "tag_lag_0"])
        .group_by("doc_id")
        .agg(
            pl.concat_list(
                [f"tok_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("span_tok"),
            pl.concat_list(
                [f"tag_lag_{i}" for i in range(-preceding, following + 1)]
                ).alias("span_tag")
            )
        .explode(["span_tok", "span_tag"])
        .with_columns(
                pre_node_tok=pl.col("span_tok").list.head(preceding),
                pre_node_tag=pl.col("span_tag").list.head(preceding)
            )
        .with_columns(
                post_node_tok=pl.col("span_tok").list.tail(following),
                post_node_tag=pl.col("span_tag").list.tail(following)
            )
        .drop(["span_tok", "span_tag"])
        .with_columns(
            Token=pl.col("pre_node_tok").list.concat("post_node_tok"),
            Tag=pl.col("pre_node_tag").list.concat("post_node_tag")
            )
        .select(["Token", "Tag"])
        .explode(["Token", "Tag"])
        .group_by(["Token", "Tag"]).len(name="Freq_Span")
        .sort("Freq_Span")
        .join(total_df, on=["Token", "Tag"])
        .with_columns(
            MI=mi_funct
            )
        .rename({"Freq_Span": "Freq Span", "Freq_Total": "Freq Total"})
        .sort("MI", "Token", descending=[True, False])
    )

    return coll_df


def keyness_table(target_frequencies: pl.DataFrame,
                  reference_frequencies: pl.DataFrame,
                  correct=False,
                  tags_only=False,
                  swap_target=False,
                  threshold=.01):
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
    if not tags_only:
        validation = OrderedDict([('Token', pl.String),
                                  ('Tag', pl.String),
                                  ('AF', pl.UInt32),
                                  ('RF', pl.Float64),
                                  ('Range', pl.Float64)])
        if (
            target_frequencies.collect_schema() != validation
            or reference_frequencies.collect_schema() != validation
        ):
            raise ValueError("""
                            Invalid DataFrame.
                            Expected DataFrames produced by frequency_table.
                            """)
    if tags_only:
        validation = OrderedDict([('Tag', pl.String),
                                  ('AF', pl.UInt32),
                                  ('RF', pl.Float64),
                                  ('Range', pl.Float64)])
        if (
            target_frequencies.collect_schema() != validation
            or reference_frequencies.collect_schema() != validation
        ):
            raise ValueError("""
                            Invalid DataFrame.
                            Expected DataFrames produced by tags_table.
                            """)

    total_target = target_frequencies.get_column("AF").sum()
    total_reference = reference_frequencies.get_column("AF").sum()
    total_tokens = total_target + total_reference

    if not correct:
        correction_tar = pl.col("AF")
        correction_ref = pl.col("AF_Ref")
    if correct:
        correction_tar = pl.col("AF").sub(
            .5 * pl.col("AF").sub(
                (pl.col("AF").add(
                    pl.col("AF_Ref")
                    ).mul(total_target / total_tokens))
                ).abs().truediv(pl.col("AF").sub((pl.col("AF").add(
                    pl.col("AF_Ref")
                    ).mul(total_target / total_tokens))))
            )
        correction_ref = pl.col("AF_Ref").add(
            .5 * pl.col("AF").sub(
                (pl.col("AF").add(
                    pl.col("AF_Ref")
                    ).mul(total_target / total_tokens))
                ).abs().truediv(pl.col("AF").sub((pl.col("AF").add(
                    pl.col("AF_Ref")
                    ).mul(total_target / total_tokens))))
            )

    if not tags_only:
        kw_df = target_frequencies.join(
            reference_frequencies, on=["Token", "Tag"],
            how="full",
            coalesce=True,
            suffix="_Ref"
            ).fill_null(strategy="zero")
    if tags_only:
        kw_df = target_frequencies.join(
            reference_frequencies, on="Tag",
            how="full",
            coalesce=True,
            suffix="_Ref"
            ).fill_null(strategy="zero")

    kw_df = (
        kw_df
        .with_columns(
            pl.when(
                pl.col("AF")
                .sub((pl.col("AF")
                      .add(pl.col("AF_Ref"))
                      .mul(total_target / total_tokens))).abs() > .25
                    )
            .then(correction_tar)
            .otherwise(pl.col("AF"))
            .alias("AF_Yates")
            )
        .with_columns(
            pl.when(
                pl.col("AF")
                .sub((pl.col("AF")
                      .add(pl.col("AF_Ref"))
                      .mul(total_target / total_tokens))).abs() > .25
                    )
            .then(correction_ref)
            .otherwise(pl.col("AF_Ref"))
            .alias("AF_Ref_Yates")
            )
        .with_columns(
            pl.when(pl.col("AF_Yates") > 0)
            .then(
                pl.col("AF_Yates")
                .mul(pl.col("AF_Yates")
                     .truediv(pl.col("AF_Yates")
                              .add(pl.col("AF_Ref"))
                              .mul(total_target / total_tokens)).log())
                              )
            .otherwise(0)
            .alias("L1")
            )
        .with_columns(
            pl.when(pl.col("AF_Ref_Yates") > 0)
            .then(
                pl.col("AF_Ref_Yates")
                .mul(pl.col("AF_Ref_Yates")
                     .truediv(pl.col("AF_Yates")
                              .add(pl.col("AF_Ref_Yates"))
                              .mul(total_reference / total_tokens)).log())
                              )
            .otherwise(0)
            .alias("L2")
            )
        .with_columns(
            pl.when(pl.col("RF") > pl.col("RF_Ref"))
            .then(
                pl.col("L1")
                .add(pl.col("L2")).mul(2).abs()
            )
            .otherwise(
                pl.col("L1")
                .add(pl.col("L2")).mul(2).abs().neg()
            )
            .alias("LL")
        )
        .with_columns(
            pl.when(pl.col("AF_Ref") == 0)
            .then(
                pl.col("AF")
                .truediv(total_target)
                .truediv(.5 / total_reference).log(base=2)
            )
            .when(pl.col("AF") == 0)
            .then(
                pl.col("AF_Ref")
                .truediv(total_reference)
                .truediv(.5 / total_target).log(base=2).neg()
            )
            .otherwise(
                pl.col("AF")
                .truediv(total_target)
                .truediv(pl.col("AF_Ref")
                         .truediv(total_reference)).log(base=2)
            )
            .alias("LR")
        )
        .with_columns(
            pl.col("LL").abs().map_elements(lambda x: chi2.sf(x, 1),
                                            return_dtype=pl.Float64)
            .alias("PV")
        )
        .sort("LL", descending=True)
        .filter(pl.col("PV") < threshold)
    )

    if not swap_target:
        kw_df = (
            kw_df
            .filter(pl.col("LL") > 0)
        )
    if swap_target:
        kw_df = (
            kw_df
            .with_columns(pl.col(
                ["LL", "LR"]
                ).mul(-1))
            .sort("LL", descending=True)
            .filter(pl.col("LL") > 0)
        )

    if not tags_only:
        return kw_df.select(["Token", "Tag", "LL", "LR", "PV",
                             "RF", "RF_Ref", "AF", "AF_Ref",
                             "Range", "Range_Ref"])

    if tags_only:
        return kw_df.select(["Tag", "LL", "LR", "PV",
                             "RF", "RF_Ref", "AF", "AF_Ref",
                             "Range", "Range_Ref"])


def tag_ruler(tokens_table: pl.DataFrame,
              doc_id: Union[str, int],
              count_by='pos') -> pl.DataFrame:
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
    count_types = ['pos', 'ds']
    if count_by not in count_types:
        raise ValueError("""
                         Invalid count_by type. Expected one of: %s
                         """ % count_types)

    validation = OrderedDict([('doc_id', pl.String),
                              ('token', pl.String),
                              ('pos_tag', pl.String),
                              ('ds_tag', pl.String),
                              ('pos_id', pl.UInt32),
                              ('ds_id', pl.UInt32)])
    if tokens_table.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by docuscope_parse.
                         """)
    max_id = tokens_table.select(pl.col("doc_id").unique().count()).item() - 1
    if type(doc_id) is int and doc_id > max_id:
        raise ValueError("""
                    Invalid doc_id. Expected an integer < or = %s
                         """ % max_id)

    if type(doc_id) is int:
        doc = tokens_table.select(
            pl.col("doc_id").unique()
            ).to_dicts()[doc_id].get("doc_id")
    else:
        doc = doc_id

    ruler_df = (
        tokens_table
        .filter(pl.col("doc_id") == doc)
    )

    if ruler_df.height < 1:
        return ruler_df

    if count_by == 'pos':
        ruler_df = (
            ruler_df
            .select(["token", "pos_tag"])
            .rename({"token": "Token", "pos_tag": "Tag"})
        )
    else:
        ruler_df = (
            ruler_df
            .select(["token", "ds_tag"])
            .rename({"token": "Token", "ds_tag": "Tag"})
         )
    ruler_df = (
            ruler_df
            .with_columns(
                pl.col("Token").str.len_chars()
                .alias("tag_end")
                )
            .with_columns(
                pl.col("tag_end").shift(1, fill_value=0).cum_sum()
                .alias("tag_start")
                )
            .with_columns(
                pl.col("tag_end").cum_sum()
                )
            .with_columns(
                pl.when(pl.col("Token").str.contains("\\s$"))
                .then(
                    pl.col("tag_end").sub(1)
                    )
                .otherwise(pl.col("tag_end"))
                )
            .select(["Token", "Tag", "tag_start", "tag_end"])
            )
    return ruler_df
