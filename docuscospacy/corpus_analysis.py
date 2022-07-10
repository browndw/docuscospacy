from tmtoolkit.corpus import doc_tokens, kwic
from tmtoolkit.tokenseq import pmi, pmi2, pmi3, index_windows_around_matches
import numpy as np
import pandas as pd
import re
from scipy.stats.distributions import chi2
from collections import Counter

import docuscospacy.corpus_utils

def convert_corpus(tm_corpus):
    """
    A simple wrapper for coverting a tmtoolkit corpus in an nltk-like list of tuples.
    
    :param tm_corpus: A tmtoolkit corpus
    :return: a list of tuples
    """
    docs = doc_tokens(tm_corpus, with_attr=True)
    tp = corpus_utils._convert_totuple(docs)
    return(tp)
 
def frequency_table(tok, n_tokens, count_by='pos'):
    """
    Generate a count of token frequencies.
    
    :param tok: A list of tuples as generated by the convert_corpus function
    :param n_tokens: A count of total tokens agaist which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe
    """
    if count_by == 'pos':
        tc = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        tc = corpus_utils._merge_ds(tok)
    phrase_range = []
    for i in range(0,len(tc)):
        phrase_range.append(list(set(tc[i])))
    phrase_range = [x for xs in phrase_range for x in xs]
    phrase_range = Counter(phrase_range)
    phrase_range = sorted(phrase_range.items(), key=lambda pair: pair[0], reverse=False)
    phrase_list = [x for xs in tc for x in xs]
    phrase_list = Counter(phrase_list)
    phrase_list = sorted(phrase_list.items(), key=lambda pair: pair[0], reverse=False)
    phrases = [x[0] for x in phrase_list]
    tags = [x[1] for x in phrases]
    if count_by == 'ds':
        tags = np.array([re.sub(r'([a-z])([A-Z])', '\\1 \\2', x) for x in tags])
    else:
        tags = np.array(tags)
    phrases = np.array([x[0] for x in phrases])
    phrase_freq = np.array([x[1] for x in phrase_list])
    phrase_prop = np.array(phrase_freq)/n_tokens*1000000
    phrase_range = np.array([x[1] for x in phrase_range])/len(tok)*100
    phrase_range = phrase_range.round(decimals=2)
    phrase_counts = list(zip(phrases.tolist(), tags.tolist(), phrase_freq.tolist(), phrase_prop.tolist(), phrase_range.tolist()))
    phrase_counts = pd.DataFrame(phrase_counts, columns=['Token', 'Tag', 'AF', 'RF', 'Range'])
    return(phrase_counts)

def tags_table(tok, n_tokens, count_by='pos'):
    """
    Generate a count of tag frequencies.
    
    :param tok: A list of tuples as generated by the convert_corpus function
    :param n_tokens: A count of total tokens agaist which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe
    """
    if count_by == 'pos':
        tc = corpus_utils._count_tags(tok, n_tokens)
    if count_by == 'ds':
        tc = corpus_utils._count_ds(tok, n_tokens)
    tag_counts = pd.DataFrame(tc, columns=['Tag', 'AF', 'RF', 'Range'])
    return(tag_counts)

def ngrams_table(tok, ng_span, n_tokens, count_by='pos'):
    """
    Generate a table of ngram frequencies.
    
    :param tok: A list of tuples as generated by the convert_corpus function
    :param ng_span: An interger between 2 and 5 representing the size of the ngrams
    :param n_tokens: A count of total tokens agaist which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe
    """
    # set limit on the size of the ngrams
    if ng_span < 2 or ng_span > 5:
        raise ValueError("Span must be < " + str(2) + " and > " + str(5))
    if count_by == 'pos':
        mtp = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        mtp = corpus_utils._merge_ds(tok)
    ml = []
    for i in range(0,len(mtp)):
        ml.append(list('_tag_'.join(x) for x in mtp[i]))
    tc = []
    for i in range(0,len(ml)):
        tc.append(list(_get_ngrams([x for x in ml[i]], n=ng_span)))
    ngram_range = []
    for i in range(0,len(tc)):
        ngram_range.append(list(set(tc[i])))
    ngram_range = [x for xs in ngram_range for x in xs]
    ngram_range = Counter(ngram_range)
    ngram_range = sorted(ngram_range.items(), key=lambda pair: pair[0], reverse=False)
    ngram_list = [x for xs in tc for x in xs]
    ngram_list = Counter(ngram_list)
    ngram_list = sorted(ngram_list.items(), key=lambda pair: pair[0], reverse=False)
    ngrams = [x[0] for x in ngram_list]
    ngrams = [sum([x[i].split('_tag_') for i in range(ng_span)], []) for x in ngrams]
    order = list(range(0, ng_span*2, 2)) + list(range(1, ng_span*2 + 1, 2))
    for l in reversed(range(len(ngrams))):
        ngrams[l] = [ngrams[l][j] for j in order]
    ngrams = np.array(ngrams)
    ngram_freq = np.array([x[1] for x in ngram_list])
    # total_ngrams = sum(ngram_freq)
    # Note: using non_punct for normalization
    ngram_prop = np.array(ngram_freq)/n_tokens*1000000
    ngram_range = np.array([x[1] for x in ngram_range])/len(tp)*100
    counts = list(zip(ngrams.tolist(), ngram_freq.tolist(), ngram_prop.tolist(), ngram_range.tolist()))
    ngram_counts = list()
    for x in counts:
        tt = tuple()
        for y in x:
            if not type(y) == list:
                tt += (y,)
            else:
                tt += (*y,)
        ngram_counts.append(tt)
    ngram_counts = pd.DataFrame(ngram_counts, columns=['Token' + str(i) for i in range (1, ng_span+1)] + ['Tag' + str(i) for i in range (1, ng_span+1)] + ['AF', 'RF', 'Range'])
    return(ngram_counts)

def coll_table(tok, node_word, l_span=4, r_span=4, statistic='pmi', count_by='pos', node_tag=None, tag_ignore=False):
    """
    Generate a table of collocations by association measure.
    
    :param tok: A list of tuples as generated by the convert_corpus function
    :param node_word: The token around with collocations are measured
    :param l_span: An integer between 0 and 9 representing the span to the left of the node word
    :param r_span: An integer between 0 and 9 representing the span to the right of the node word
    :param statistic: The association measure to be calculated. One of: 'pmi', 'npmi', 'pmi2', 'pmi3'
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :param node_tag: A value specifying the tag of the node word. If the node_word were 'can', a node_tag 'V' would search for can as a verb.
    :return: a dataframe
    """
    stats = {'pmi', 'npmi', 'pmi2', 'pmi3'}
    if statistic not in stats:
        raise ValueError("results: statistic must be one of %r." % stats)
    if l_span < 0 or l_span > 9:
        raise ValueError("Span must be < " + str(0) + " and > " + str(9))
    if r_span < 0 or r_span > 9:
        raise ValueError("Span must be < " + str(0) + " and > " + str(9))
    if bool(tag_ignore) == True:
        node_tag = None
    if count_by == 'pos':
        tc = corpus_utils._merge_tags(tok)
    if count_by == 'ds':
        tc = corpus_utils._merge_ds(tok)
    in_span = []
    for i in range(0,len(tc)):
        tpf = tc[i]
        # create a boolean vector for node word
        if node_tag is None:
            v = [t[0] == node_word for t in tpf]
        else:
            v = [t[0] == node_word and t[1].startswith(node_tag) for t in tpf]
        if sum(v) > 0:
            # get indices within window around the node
            idx = list(index_windows_around_matches(np.array(v), left=l_span, right=r_span, flatten=False))
            # remove node word from collocates
            idx = np.delete(idx, l_span, axis=1)
            idx = [x for xs in idx for x in xs]
            coll = [tpf[i] for i in idx]
        else:
            coll = []
        in_span.append(coll)
    in_span = [x for xs in in_span for x in xs]
    tc = [x for xs in tc for x in xs]
    df_total = pd.DataFrame(tc, columns=['token', 'tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.drop(columns=['tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.groupby(['token']).value_counts().to_frame('total_freq').reset_index()
    else:
        df_total = df_total.groupby(['token','tag']).value_counts().to_frame('total_freq').reset_index()
    df_span = pd.DataFrame(in_span, columns=['token', 'tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.drop(columns=['tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.groupby(['token']).value_counts().to_frame('span_freq').reset_index()
    else:
        df_span = df_span.groupby(['token','tag']).value_counts().to_frame('span_freq').reset_index()
    if node_tag is None:
        node_freq = sum(df_total[df_total['token'] == node_word]['total_freq'])
    else:
        node_freq = sum(df_total[(df_total['token'] == node_word) & (df_total['tag'].str.startswith(node_tag, na=False))]['total_freq'])
    if bool(tag_ignore) == True:
        df = pd.merge(df_span, df_total, how='inner', on=['token'])
    else:
        df = pd.merge(df_span, df_total, how='inner', on=['token', 'tag'])
    if statistic=='pmi':
        df['MI'] = np.vectorize(pmi)(node_freq, df['total_freq'], df['span_freq'], sum(df_total['total_freq']), normalize=False)
    if statistic=='npmi':
        df['MI'] = np.vectorize(pmi)(node_freq, df['total_freq'], df['span_freq'], sum(df_total['total_freq']), normalize=True)
    if statistic=='pmi2':
        df['MI'] = np.vectorize(pmi2)(node_freq, df['total_freq'], df['span_freq'], sum(df_total['total_freq']))
    if statistic=='pmi3':
        df['MI'] = np.vectorize(pmi3)(node_freq, df['total_freq'], df['span_freq'], sum(df_total['total_freq']))
    return(df)

def kwic_center_node(tm_corpus, node_word,  ignore_case=True, glob=False):
    """
    Generate a KWIC table with the node word in the center column.
    
    :param tm_corpus: A tmtoolkit corpus
    :param node_word: The token of interest
    :param ignore_case: If set to False, search will be case sensitive
    :param glob: If set to True, glob-style searching is enabled
    :return: a dataframe
    """
    if bool(glob)==False:
        kl = kwic(tm_corpus, node_word, context_size=10, ignore_case=ignore_case)
    else:
        kl = kwic(tm_corpus, node_word, context_size=10, ignore_case=ignore_case, match_type='glob')
    keys = [k for k in kl.keys() for v in kl[k]]
    token_list = [v for k in kl.keys() for v in kl[k]]
    pre_node = [' '.join(l[:10]) for l in token_list]
    # set a span after which characters are trimmed for display
    pre_node = [('..' + l[len(l)-75:]) if len(l) > 75 else l for l in pre_node]
    node = [l[10] for l in token_list]
    post_node = [' '.join(l[11:]) for l in token_list]
    # apply same trim span
    post_node = [(l[:75] + '..') if len(l) > 75 else l for l in post_node]
    df = pd.DataFrame.from_dict({'Doc': keys, 'Pre-Node': pre_node, 'Node': node, 'Post-Node': post_node})
    return(df)


def keyness_table(target_counts, ref_counts, total_target, total_reference, correct=False, tags_only=False):
    """
    Generate a keyness table comparing token frequencies from a taget and a reference corpus
    
    :param target_counts: A frequency table from a target corpus
    :param ref_counts: A frequency table from a reference corpus
    :param total_target: Total number of tokens in the target corpus
    :param total_target: Total number of tokens in the reference corpus
    :param correct: If True, apply the Yates correction to the log-likelihood calculation
    :param tags_only: If True, it is assumed the frequency tables are of the type produce by the tags_table function
    :return: a dataframe
    """
    if bool(tags_only) == True:
        target_counts.columns=['Tag','AF', 'RF', 'Range']
    else:
        target_counts.columns=['Token', 'Tag','AF', 'RF', 'Range']
    if bool(tags_only) == True:
        ref_counts.columns=['Tag', 'AF Ref', 'RF Ref', 'Range Ref']
    else:
        ref_counts.columns=['Token', 'Tag', 'AF Ref', 'RF Ref', 'Range Ref']
    if bool(tags_only) == True:
        df = pd.merge(target_counts, ref_counts, how='outer', on=['Tag'])
    else:
        df = pd.merge(target_counts, ref_counts, how='outer', on=['Token', 'Tag'])
    df.fillna(0, inplace=True)
    if bool(correct) == True:
        df['LL'] = np.vectorize(_log_like)(df['AF'], df['AF Ref'], total_target, total_reference, correct=True)
    else:
        df['LL'] = np.vectorize(_log_like)(df['AF'], df['AF Ref'], total_target, total_reference, correct=False)
    df['LR'] = np.vectorize(_log_ratio)(df['AF'], df['AF Ref'], total_target, total_reference)
    df['PV'] = chi2.sf(df['LL'], 1)
    df.PV = df.PV.round(5)
    return(df)
