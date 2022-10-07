"""
Functions for analyzing corpus data tagged with DocuScope and CLAWS7.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

from tmtoolkit.corpus import doc_tokens, kwic
from tmtoolkit.tokenseq import pmi, pmi2, pmi3, index_windows_around_matches
import numpy as np
import pandas as pd
import re
from scipy.stats.distributions import chi2
from collections import Counter

from .corpus_utils import _convert_totuple, _merge_tags, _merge_ds, _count_tags, _count_ds, _log_like, _log_ratio, _get_ngrams, _groupby_consecutive, _conlltags2tree, Tree

def convert_corpus(tm_corpus):
    """
    A simple wrapper for coverting a tmtoolkit corpus in an nltk-like dictionary of tuples.
    
    :param tm_corpus: A tmtoolkit corpus
    :return: a dictionary of tuples
    """
    docs = doc_tokens(tm_corpus, with_attr=['token', 'whitespace', 'ent_iob', 'ent_type', 'tag'])
    tp = _convert_totuple(docs)
    d = {tm_corpus.doc_labels[i]: tp[i] for i in range(0,len(tp))}
    return(d)
 
def frequency_table(tok, n_tokens, count_by='pos'):
    """
    Generate a count of token frequencies.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param n_tokens: A count of total tokens against which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe of absolute frequencies, normalized frequencies (per million tokens) and ranges
    """
    tok = list(tok.values())
    if count_by == 'pos':
        tc = _merge_tags(tok)
    if count_by == 'ds':
        tc = _merge_ds(tok)
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
    phrase_counts.sort_values(by=['AF', 'Token'], ascending=[False, True], inplace=True)
    phrase_counts.reset_index(drop=True, inplace=True)
    return(phrase_counts)

def tags_table(tok, n_tokens, count_by='pos'):
    """
    Generate a count of tag frequencies.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param n_tokens: A count of total tokens against which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe of absolute frequencies, normalized frequencies (per million tokens) and ranges
    """
    tok = list(tok.values())
    if count_by == 'pos':
        tc = _count_tags(tok, n_tokens)
    if count_by == 'ds':
        tc = _count_ds(tok, n_tokens)
    tag_counts = pd.DataFrame(tc, columns=['Tag', 'AF', 'RF', 'Range'])
    tag_counts.sort_values(by=['AF', 'Tag'], ascending=[False, True], inplace=True)
    tag_counts.reset_index(drop=True, inplace=True)
    return(tag_counts)

def tags_dtm(tok, count_by='pos'):
    """
    Generate a document-term matrix of raw tag counts.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe of absolute tag frequencies for each document
    """
    doc_id = list(tok.keys())
    tok = list(tok.values())
    if count_by == 'pos':
        tc = _merge_tags(tok)
    if count_by == 'ds':
        tc = _merge_ds(tok)
    remove_starttags = ('Y', 'FU')
    tag_list = []
    for i in range(0,len(tc)):
        tags = [x[1] for x in tc[i]]
        if count_by == 'pos':
            tags = [x for x in tags if not x.startswith(remove_starttags)]
        tag_list.append(tags)
    tag_counts = []
    for i in range(0,len(tag_list)):
        counts = Counter(tag_list[i])
        tag_counts.append(counts)
    df = pd.DataFrame.from_records(tag_counts)
    df = df.fillna(0)
    df = df.reindex(sorted(df.columns), axis=1)
    df.insert(0, 'doc_id', doc_id)
    return(df)

def ngrams_table(tok, ng_span, n_tokens, count_by='pos'):
    """
    Generate a table of ngram frequencies.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param ng_span: An interger between 2 and 5 representing the size of the ngrams
    :param n_tokens: A count of total tokens against which to normalize
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: a dataframe containing a token sequence the length of the span, a tag sequence the length of the span, absolute frequencies, normalized frequencies (per million tokens) and ranges
    """
    tok = list(tok.values())
    # set limit on the size of the ngrams
    if ng_span < 2 or ng_span > 5:
        raise ValueError("Span must be < " + str(2) + " and > " + str(5))
    if count_by == 'pos':
        mtp = _merge_tags(tok)
    if count_by == 'ds':
        mtp = _merge_ds(tok)
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
    ngram_range = np.array([x[1] for x in ngram_range])/len(tok)*100
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
    ngram_counts.sort_values(by=['AF', 'Token1'], ascending=[False, True], inplace=True)
    ngram_counts.reset_index(drop=True, inplace=True)
    return(ngram_counts)

def coll_table(tok, node_word, l_span=4, r_span=4, statistic='pmi', count_by='pos', node_tag=None, tag_ignore=False):
    """
    Generate a table of collocations by association measure.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param node_word: The token around with collocations are measured
    :param l_span: An integer between 0 and 9 representing the span to the left of the node word
    :param r_span: An integer between 0 and 9 representing the span to the right of the node word
    :param statistic: The association measure to be calculated. One of: 'pmi', 'npmi', 'pmi2', 'pmi3'
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :param node_tag: A value specifying the tag of the node word. If the node_word were 'can', a node_tag 'V' would search for can as a verb.
    :param tag_ignore: A boolean value indicating whether or not tags should be ignored during analysis.
    :return: a dataframe containing collocate tokens, tags, the absolute frequency the collocate in the corpus, the absolute frequency of the collocate within the designated span, and the association measure.
    """
    tok = list(tok.values())
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
        tc = _merge_tags(tok)
    if count_by == 'ds':
        tc = _merge_ds(tok)
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
            node_idx = [i for i, x in enumerate(v) if x == True]
            # remove node word from collocates
            coll_idx = [np.setdiff1d(idx[i], node_idx[i]) for i in range(len(idx))]
            coll_idx = [x for xs in coll_idx for x in xs]
            coll = [tpf[i] for i in coll_idx]
        else:
            coll = []
        in_span.append(coll)
    in_span = [x for xs in in_span for x in xs]
    tc = [x for xs in tc for x in xs]
    df_total = pd.DataFrame(tc, columns=['Token', 'Tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.drop(columns=['Tag'])
    if bool(tag_ignore) == True:
        df_total = df_total.groupby(['Token']).value_counts().to_frame('Freq Total').reset_index()
    else:
        df_total = df_total.groupby(['Token','Tag']).value_counts().to_frame('Freq Total').reset_index()
    df_span = pd.DataFrame(in_span, columns=['Token', 'Tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.drop(columns=['Tag'])
    if bool(tag_ignore) == True:
        df_span = df_span.groupby(['Token']).value_counts().to_frame('Freq Span').reset_index()
    else:
        df_span = df_span.groupby(['Token','Tag']).value_counts().to_frame('Freq Span').reset_index()
    if node_tag is None:
        node_freq = sum(df_total[df_total['Token'] == node_word]['Freq Total'])
    else:
        node_freq = sum(df_total[(df_total['Token'] == node_word) & (df_total['Tag'].str.startswith(node_tag, na=False))]['Freq Total'])
    if bool(tag_ignore) == True:
        df = pd.merge(df_span, df_total, how='inner', on=['Token'])
    else:
        df = pd.merge(df_span, df_total, how='inner', on=['Token', 'Tag'])
    if statistic=='pmi':
        df['MI'] = pmi(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']), normalize=False)
    if statistic=='npmi':
        df['MI'] = pmi(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']), normalize=True)
    if statistic=='pmi2':
        df['MI'] = pmi2(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']))
    if statistic=='pmi3':
        df['MI'] = pmi3(node_freq, df['Freq Total'], df['Freq Span'], sum(df_total['Freq Total']))
    df.sort_values(by=['MI', 'Token'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

def kwic_center_node(tm_corpus, node_word,  ignore_case=True, glob=False):
    """
    Generate a KWIC table with the node word in the center column.
    
    :param tm_corpus: A tmtoolkit corpus
    :param node_word: The token of interest
    :param ignore_case: If set to False, search will be case sensitive
    :param glob: If set to True, glob-style searching is enabled
    :return: a dataframe with the node word in a center column and context columns on either side.
    """
    if bool(glob)==False:
        kl = kwic(tm_corpus, node_word, context_size=10, ignore_case=ignore_case, highlight_keyword="##")
    else:
        kl = kwic(tm_corpus, node_word, context_size=10, ignore_case=ignore_case, match_type='glob', highlight_keyword="##")
    keys = [k for k in kl.keys() for v in kl[k]]
    token_list = [v for k in kl.keys() for v in kl[k]]
    node_idx = [i for x in range(len(token_list)) for i in range(len(token_list[x])) if token_list[x][i].startswith('##') and token_list[x][i].endswith('##') and len(token_list[x][i]) > 2]
    pre_node = [' '.join(token_list[i][:node_idx[i]]) for i in range(len(token_list))]
    # set a span after which characters are trimmed for display
    pre_node = [('..' + l[len(l)-75:]) if len(l) > 75 else l for l in pre_node]
    node = [token_list[i][node_idx[i]].replace('##', '') for i in range(len(token_list))]
    post_node = [' '.join(token_list[i][node_idx[i] + 1:]) for i in range(len(token_list))]
    # apply same trim span
    post_node = [(l[:75] + '..') if len(l) > 75 else l for l in post_node]
    df = pd.DataFrame.from_dict({'Doc': keys, 'Pre-Node': pre_node, 'Node': node, 'Post-Node': post_node})
    return(df)


def keyness_table(target_counts, ref_counts, correct=False, tags_only=False):
    """
    Generate a keyness table comparing token frequencies from a taget and a reference corpus
    
    :param target_counts: A frequency table from a target corpus
    :param ref_counts: A frequency table from a reference corpus
    :param correct: If True, apply the Yates correction to the log-likelihood calculation
    :param tags_only: If True, it is assumed the frequency tables are of the type produce by the tags_table function
    :return: a dataframe of absolute frequencies, normalized frequencies (per million tokens) and ranges for both corpora, as well as keyness values as calculated by log-likelihood and effect size as calculated by Log Ratio.
    """
    total_target = target_counts['AF'].sum()
    total_reference = ref_counts['AF'].sum()
    if bool(tags_only) == True:
        df_1 = target_counts.set_axis(['Tag','AF', 'RF', 'Range'], axis=1, inplace=False)
    else:
        df_1 = target_counts.set_axis(['Token', 'Tag','AF', 'RF', 'Range'], axis=1, inplace=False)
    if bool(tags_only) == True:
        df_2 = ref_counts.set_axis(['Tag', 'AF Ref', 'RF Ref', 'Range Ref'], axis=1, inplace=False)
    else:
        df_2 = ref_counts.set_axis(['Token', 'Tag', 'AF Ref', 'RF Ref', 'Range Ref'], axis=1, inplace=False)
    if bool(tags_only) == True:
        df = pd.merge(df_1, df_2, how='outer', on=['Tag'])
    else:
        df = pd.merge(df_1, df_2, how='outer', on=['Token', 'Tag'])
    df.fillna(0, inplace=True)
    if bool(correct) == True:
        df['LL'] = np.vectorize(_log_like)(df['AF'], df['AF Ref'], total_target, total_reference, correct=True)
    else:
        df['LL'] = np.vectorize(_log_like)(df['AF'], df['AF Ref'], total_target, total_reference, correct=False)
    df['LR'] = np.vectorize(_log_ratio)(df['AF'], df['AF Ref'], total_target, total_reference)
    df['PV'] = chi2.sf(abs(df['LL']), 1)
    df.PV = df.PV.round(5)
    if bool(tags_only) == True:
        df = df.iloc[:, [0,7,8,9,1,2,3,4,5,6]]
        df.sort_values(by='LL', ascending=False, inplace=True)
    else:
        df = df.iloc[:, [0,1,8,9,10,2,3,4,5,6,7]]
        df.sort_values(by=['LL', 'Token'], ascending=[False, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return(df)

def tag_ruler(tok, key, count_by='pos'):
    """
    Retrieve spans of tags to facitiatve tag highligting in a single text.
    
    :param tok: A dictionary of tuples as generated by the convert_corpus function
    :param key: A document name as stored as one of the keys in the tok dictionary
    :param count_by: One of 'pos' or 'ds' for aggregating tokens
    :return: A dataframe including all tokens, tags, tags start index, and tag end index
    """
    tpf = list(tok.get(key))
    if count_by == 'ds':
        token_tp = tpf
    else:
        token_list = [x[0] for x in tpf]
        tag_list = [x[1] for x in tpf]
        tag_seq = [re.findall(r'\d\d$', x) for x in tag_list]
        tag_seq = [x for sublist in tag_seq for x in (sublist or ['99'])]
        tag_seq = [int(x) for x in tag_seq]
        tag_seq = list(_groupby_consecutive(lst=tag_seq))
        for x in tag_seq:
            x[0] = re.sub('\\d+', 'B-', str(x[0]))
        tag_seq = [x for xs in tag_seq for x in xs]
        tag_seq = ['I-' if isinstance(x, int) else x for x in tag_seq]
        tag_seq = [a_+str(b_) for a_,b_ in zip(tag_seq, tag_list)]
        tag_seq = [re.sub(r'\d\d$', '', x) for x in tag_seq]
        token_tp = list(zip(token_list, tag_list, tag_seq))
    ne_tree = _conlltags2tree(token_tp)
    agg_tokens = []
    for subtree in ne_tree:
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = "".join([token for token, pos in subtree.leaves()])
        else:
            original_label = 'Untagged'
            original_string = subtree[0]
        agg_tokens.append((original_string, original_label))
    df = pd.DataFrame(agg_tokens, columns=['Token', 'Tag'])
    tag_len = list(df['Token'].str.len())
    tag_start = [0] + tag_len[:-1]
    tag_start = np.cumsum(tag_start)
    tag_end = np.cumsum(tag_len)
    s = re.compile('\s$')    
    s_final = [bool(s.search(x)) for x in df['Token']]
    tag_end = np.where(np.array(s_final)==True, tag_end-1, tag_end)
    df = df.assign(tag_start = tag_start,
                   tag_end = tag_end)
    return(df)
