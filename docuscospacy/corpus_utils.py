"""
Misc. utility functions for corpus processing.

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

import string
import re
import numpy as np
from functools import partial
from itertools import groupby, islice
from collections import Counter
from operator import itemgetter
from typing import Union, List, Callable, Optional

def _convert_totuple(tok):
    """
    Convert a tmtoolkit tokens object to a tuple of the nltk-type: [(token), (tag), (iob-ent)].
    
    :param tok: a tmtoolkit tokens object
    """
    token_tuple = []
    is_punct = re.compile("[{}]+\s*$".format(re.escape(string.punctuation)))
    is_digit = re.compile("\d[\d{}]*\s*$".format(re.escape(string.punctuation)))
    for i in range(0,len(tok)):
        token_list = list(tok.values())[i]['token']
        ws_list = list(tok.values())[i]['whitespace']
        token_list = list(map(''.join, zip(token_list, ws_list)))
        iob_list = list(tok.values())[i]['ent_iob']
        iob_list = [x.replace('IS_DIGIT','B') for x in iob_list]
        iob_list = [x.replace('IS_ALPHA','I') for x in iob_list]
        iob_list = [x.replace('IS_ASCII','O') for x in iob_list]
        ent_list = list(tok.values())[i]['ent_type']
        iob_ent = list(map('-'.join, zip(iob_list, ent_list)))
        tag_list = list(tok.values())[i]['tag']
        # correct some mistagging
        tag_list = ['Y' if bool(is_punct.match(token_list[i])) else v for i, v in enumerate(tag_list)]
        tag_list = ['MC' if bool(is_digit.match(token_list[i])) and tag_list[i] != 'Y' else v for i, v in enumerate(tag_list)]
        token_tuple.append(list(zip(token_list, tag_list, iob_ent)))
    return(token_tuple)

def _groupby_consecutive(lst):
    """
    Convenience function for grouping consecutive items in a list.
    
    :param lst: a list
    """
    for _, g in groupby(enumerate(lst), lambda x: x[0] - x[1]):
        yield list(map(itemgetter(1), g))


# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/tokenseq.py
def _index_windows_around_matches(matches: np.ndarray, left: int, right: int,
                                 flatten: bool = False, remove_overlaps: bool = True) \
        -> Union[List[List[int]], np.ndarray]:
    """
    Take a boolean 1D array `matches` of length N and generate an array of indices, where each occurrence of a True
    value in the boolean vector at index i generates a sequence of the form:

    .. code-block:: text

        [i-left, i-left+1, ..., i, ..., i+right-1, i+right, i+right+1]

    If `flatten` is True, then a flattened NumPy 1D array is returned. Otherwise, a list of NumPy arrays is returned,
    where each array contains the window indices.

    `remove_overlaps` is only applied when `flatten` is True.

    Example with ``left=1 and right=1, flatten=False``:

    .. code-block:: text

        input:
        #   0      1      2      3     4      5      6      7     8
        [True, True, False, False, True, False, False, False, True]
        output (matches *highlighted*):
        [[0, *1*], [0, *1*, 2], [3, *4*, 5], [7, *8*]]

    Example with ``left=1 and right=1, flatten=True, remove_overlaps=True``:

    .. code-block:: text

        input:
        #   0      1      2      3     4      5      6      7     8
        [True, True, False, False, True, False, False, False, True]
        output (matches *highlighted*, other values belong to the respective "windows"):
        [*0*, *1*, 2, 3, *4*, 5, 7, *8*]
    """
    if not isinstance(matches, np.ndarray) or matches.dtype != bool:
        raise ValueError('`matches` must be a boolean NumPy array')
    if not isinstance(left, int) or left < 0:
        raise ValueError('`left` must be an integer >= 0')
    if not isinstance(right, int) or right < 0:
        raise ValueError('`right` must be an integer >= 0')

    ind = np.where(matches)[0]
    nested_ind = list(map(lambda x: np.arange(x - left, x + right + 1), ind))

    if flatten:
        if not nested_ind:
            return np.array([], dtype=int)

        window_ind = np.concatenate(nested_ind)
        window_ind = window_ind[(window_ind >= 0) & (window_ind < len(matches))]

        if remove_overlaps:
            return np.sort(np.unique(window_ind))
        else:
            return window_ind
    else:
        return [w[(w >= 0) & (w < len(matches))] for w in nested_ind]

def _merge_tags(tok):
    """
    Merge part-of-speech tag sequences into a single token like 'for example' or 'in spite of'.
    
    :param tok: a tokens tuple object
    """
    tok = [[(word.lower(), tag, ds) for word, tag, ds in element] for element in tok]
    tok = [[(word.strip(), tag, ds) for word, tag, ds in element] for element in tok]
    p = re.compile('[a-z]')
    phrase_list = []
    for i in range(0,len(tok)):
        # filter out strings that don't contain at least one alphabetic character
        tpf = [x for x in tok[i] if p.search(x[0])]
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
                original_string = " ".join([token for token, pos in subtree.leaves()])
            else:
                original_label = 'Untagged'
                original_string = subtree[0]
            agg_tokens.append((original_string, original_label))
        phrase_list.append(agg_tokens)
    return(phrase_list)

def _merge_ds(tok):
    """
    Merge DocuScope NER sequences into a single token.
    
    :param tok: a tokens tuple object
    """
    tok = [[(word.lower(), tag, ds) for word, tag, ds in element] for element in tok]
    tok = [[(word.strip(), tag, ds) for word, tag, ds in element] for element in tok]
    p = re.compile('[a-z]')
    phrase_list = []
    for i in range(0,len(tok)):
        # filter out strings that don't contain at least one alphabetic character
        tpf = [x for x in tok[i] if not (x[2] == 'O-' and bool(p.search(x[0])) == False)]
        ne_tree = _conlltags2tree(tpf)
        agg_tokens = []
        for subtree in ne_tree:
            if type(subtree) == Tree:
                original_label = subtree.label()
                original_string = " ".join([token for token, pos in subtree.leaves()])
            else:
                original_label = 'Untagged'
                original_string = subtree[0]
            agg_tokens.append((original_string, original_label))
        phrase_list.append(agg_tokens)
    return(phrase_list)

def _count_tags(tok, n_tokens):
    """
    Count part-of-speech tags.
    
    :param tok: a tokens tuple object
    :param n_tokens: total number of tokens against which to normalize
    """
    tag_list = []
    # remove tags for puct, unidentified, and multitoken units
    remove_starttags = ('Y', 'FU')
    remove_endtags = ('22', '32', '33', '42', '43', '44')
    for i in range(0,len(tok)):
        tags = [x[1] for x in tok[i]]
        tags = [x for x in tags if not x.startswith(remove_starttags)]
        tags = [x for x in tags if not x.endswith(remove_endtags)]
        tags = [re.sub(r'\d\d', '', x) for x in tags]
        tag_list.append(tags)
    tag_range = []
    for i in range(0,len(tok)):
        tag_range.append(list(set(tag_list[i])))
    tag_range = [x for xs in tag_range for x in xs]
    tag_range = Counter(tag_range)
    tag_range = sorted(tag_range.items(), key=lambda pair: pair[0], reverse=False)
    tag_list = [x for xs in tag_list for x in xs]
    tag_list = Counter(tag_list)
    tag_list = sorted(tag_list.items(), key=lambda pair: pair[0], reverse=False)
    tags = np.array([x[0] for x in tag_list])
    tag_freq = np.array([x[1] for x in tag_list])
    tag_prop = np.array(tag_freq)/n_tokens*100
    tag_range = np.array([x[1] for x in tag_range])/len(tok)*100
    tag_counts = zip(tags.tolist(), tag_freq.tolist(), tag_prop.tolist(), tag_range.tolist())
    tag_counts = list(tag_counts)
    return(tag_counts)


def _count_ds(tok, n_tokens):
    """
    Count part-of-speech tags.
    
    :param tok: a tokens tuple object
    :param n_tokens: total number of tokens against which to normalize
    """
    ds_list = []
    for i in range(0,len(tok)):
        ds_cats = [x[2] for x in tok[i]]
        # filter for benning entity tags
        ds_cats = [x for x in ds_cats if x.startswith('B-')]
        ds_list.append(ds_cats)
    ds_range = []
    for i in range(0,len(tok)):
        ds_range.append(list(set(ds_list[i])))
    ds_range = [x for xs in ds_range for x in xs]
    ds_range = Counter(ds_range)
    ds_range = sorted(ds_range.items(), key=lambda pair: pair[0], reverse=False)
    ds_list = [x for xs in ds_list for x in xs]
    ds_list = Counter(ds_list)
    ds_list = sorted(ds_list.items(), key=lambda pair: pair[0], reverse=False)
    ds_cats = [x[0] for x in ds_list]
    ds_cats = [x.replace('B-', '') for x in ds_cats]
    ds_cats = [re.sub(r'([a-z])([A-Z])', '\\1 \\2', x) for x in ds_cats]
    ds_cats = np.array(ds_cats)
    ds_freq = np.array([x[1] for x in ds_list])
    ds_prop = np.array(ds_freq)/n_tokens*100
    ds_range = np.array([x[1] for x in ds_range])/len(tok)*100
    ds_counts = zip(ds_cats.tolist(), ds_freq.tolist(), ds_prop.tolist(), ds_range.tolist())
    ds_counts = list(ds_counts)
    return(ds_counts)


def _get_ngrams(iterable, n=2):
    """
    Helper function for splitting list into ngrams.
    
    :param iterable: a list to iterate over
    :param n: the size of the ngram to be generated
    """
    return zip(*[islice(iterable, i, None) for i in range(n)])

def _log_like(n_target, n_reference, total_target, total_reference, correct=False):
    """
    Calculate log-likelihood (or G2) hypothesis test.
    
    :param n_target: token frequency in the target corpus
    :param n_reference: token frequency in the reference corpus
    :param total_target: total tokens in the target corpus
    :param total_reference: total tokens in the reference corpus
    :param correct: if True, apply the Yates correction
    """
    expected_a = (n_target + n_reference)*(total_target/(total_target + total_reference))
    expected_b = (n_target + n_reference)*(total_reference/(total_target + total_reference))
    if bool(correct) == True:
        n_a = (n_target - 0.5) if n_target - expected_a > 0.25 else n_target
        n_b = (n_reference + 0.5) if n_target - expected_a > 0.25 else n_reference
        n_a = (n_target + 0.5) if expected_a - n_target > 0.25 else n_a
        n_b = (n_reference - 0.5) if expected_a - n_target > 0.25 else n_b
    else:
        n_a = n_target
        n_b = n_reference
    L1 = 0 if n_a == 0 else n_a*np.log(n_a/expected_a)
    L2 = 0 if n_b == 0 else n_b*np.log(n_b/expected_b)
    likelihood = 2*(L1 + L2)
    if n_target/total_target > n_reference/total_reference:
        likelihood = likelihood
    else:
        likelihood = -likelihood
    return(likelihood)


def _log_ratio(n_target, n_reference, total_target, total_reference):
    """
    Calculate Log Ratio effect size.
    
    :param n_target: token frequency in the target corpus
    :param n_reference: token frequency in the reference corpus
    :param total_target: total tokens in the target corpus
    :param total_reference: total tokens in the reference corpus
    """
    percent_a = 0.5 / total_target if n_target == 0 else n_target/total_target
    percent_b = 0.5 / total_reference if n_reference == 0 else n_reference/total_reference
    ratio = np.log2(percent_a / percent_b)
    return(ratio)


# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/tokenseq.py
def _PMI(x: np.ndarray, y: np.ndarray, xy: np.ndarray, n_total: Optional[int] = None, logfn: Callable = np.log2,
        k: int = 1, normalize: bool = False) -> np.ndarray:
    """
    Calculate pointwise mutual information measure (PMI) either from probabilities p(x), p(y), p(x, y) given as `x`,
    `y`, `xy`, or from total counts `x`, `y`, `xy` and additionally `n_total`. Setting `k` > 1 gives PMI^k variants.
    Setting `normalized` to True gives normalized PMI (NPMI) as in [Bouma2009]_. See [RoleNadif2011]_ for a comparison
    of PMI variants.

    Probabilities should be such that ``p(x, y) <= min(p(x), p(y))``.

    :param x: probabilities p(x) or count of occurrence of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrence of y (interpreted as count if `n_total` is given)
    :param xy: probabilities p(x, y) or count of occurrence of x *and* y (interpreted as count if `n_total` is given)
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same length as inputs containing (N)PMI measures for each input probability
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    if n_total is not None:
        if n_total < 1:
            raise ValueError('`n_total` must be strictly positive')
        x = x/n_total
        y = y/n_total
        xy = xy/n_total

    pmi_val = logfn(xy) - logfn(x * y)

    if k > 1:
        return pmi_val - (1-k) * logfn(xy)
    else:
        if normalize:
            return pmi_val / -logfn(xy)
        else:
            return pmi_val

_NPMI = partial(_PMI, k=1, normalize=True)
_PMI2 = partial(_PMI, k=2, normalize=False)
_PMI3 = partial(_PMI, k=3, normalize=False)

# https://github.com/WZBSocialScienceCenter/tmtoolkit/blob/master/tmtoolkit/bow/bow_stats.py
def _doc_frequencies(dtm, min_val=1, proportions=0):
    """
    For each term in the vocab of `dtm` (i.e. its columns), return how often it occurs at least `min_val` times per
    document.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param min_val: threshold for counting occurrences
    :param proportions: one of :attr:`~tmtoolkit.types.Proportion`: ``NO (0)`` – return counts; ``YES (1)`` – return
                        proportions; ``LOG (2)`` – return log of proportions
    :return: NumPy array of size M (vocab size) indicating how often each term occurs at least `min_val` times.
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    doc_freq = np.sum(dtm >= min_val, axis=0)

    if doc_freq.ndim != 1:
        doc_freq = doc_freq.A.flatten()

    if proportions == 1:
        return doc_freq / dtm.shape[0]
    elif proportions == 2:
        return np.log(doc_freq) - np.log(dtm.shape[0])
    else:
        return doc_freq

def _doc_lengths(dtm):
    """
    Return the length, i.e. number of terms for each document in document-term-matrix `dtm`.
    This corresponds to the row-wise sums in `dtm`.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw terms counts
    :return: NumPy array of size N (number of docs) with integers indicating the number of terms per document
    """
    if dtm.ndim != 2:
        raise ValueError('`dtm` must be a 2D array/matrix')

    res = np.sum(dtm, axis=1)
    if res.ndim != 1:
        return res.A.flatten()
    else:
        return res

def _tf_proportions(dtm, norm=False, scale=False):
    """
    Transform raw count document-term-matrix `dtm` to term frequency matrix with proportions, i.e. term counts
    normalized by document length.

    Note that this may introduce NaN values due to division by zero when a document is of length 0.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts
    :return: (sparse) term frequency matrix of size NxM with proportions, i.e. term counts normalized by document length
    """

    norm_factor = 1 / np.array(_doc_lengths(dtm))[:, None]   # shape: Nx1

    res = dtm * norm_factor
    
    if norm == True:
        res *=100
    else:
        res
    if scale == True:
        scaled_res = res.select_dtypes(include='number').apply(scipy.stats.zscore)
        res = pd.DataFrame(scaled_res, index=res.index, columns=res.columns)
    else:
        res
    if isinstance(res, np.matrix):
        return res.A
    else:
        return res

def _idf(dtm, smooth_log=1, smooth_df=1):
    """
    Calculate inverse document frequency (idf) vector from raw count document-term-matrix `dtm` with formula
    ``log(smooth_log + N / (smooth_df + df))``, where ``N`` is the number of documents, ``df`` is the document frequency
    (see function :func:`~tmtoolkit.bow.bow_stats.doc_frequencies`), `smooth_log` and `smooth_df` are smoothing
    constants. With default arguments, the formula is thus ``log(1 + N/(1+df))``.

    Note that this may introduce NaN values due to division by zero when a document is of length 0.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts.
    :param smooth_log: smoothing constant inside log()
    :param smooth_df: smoothing constant to add to document frequency
    :return: NumPy array of size M (vocab size) with inverse document frequency for each term in the vocab
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    n_docs = dtm.shape[0]
    df = _doc_frequencies(dtm)

    if smooth_log == smooth_df == 1:      # log1p is faster than the equivalent log(1 + x)
        # log(1 + N/(1+df)) = log((1+df+N)/(1+df)) = log(1+df+N) - log(1+df) = log1p(df+N) - log1p(df)
        return np.log1p(df + n_docs) - np.log1p(df)
    else:
        # with s = smooth_log and t = smooth_df
        # log(s + N/(t+df)) = log((s(t+df)+N)/(t+df)) = log(s(t+df)+N) - log(t+df)
        return np.log(smooth_log * (smooth_df + df) + n_docs) - np.log(smooth_df + df)


def _tfidf(dtm, tf_func=_tf_proportions, idf_func=_idf, **kwargs):
    """
    Calculate tfidf (term frequency inverse document frequency) matrix from raw count document-term-matrix `dtm` with
    matrix multiplication ``tf * diag(idf)``, where `tf` is the term frequency matrix ``tf_func(dtm)`` and ``idf`` is
    the document frequency vector ``idf_func(dtm)``.

    :param dtm: (sparse) document-term-matrix of size NxM (N docs, M is vocab size) with raw term counts
    :param tf_func: function to calculate term-frequency matrix; see ``tf_*`` functions in this module
    :param idf_func: function to calculate inverse document frequency vector; see ``tf_*`` functions in this module
    :param kwargs: additional parameters passed to `tf_func` or `idf_func` like `K` or `smooth` (depending on which
                   parameters these functions except)
    :return: (sparse) tfidf matrix of size NxM
    """
    if dtm.ndim != 2 or 0 in dtm.shape:
        raise ValueError('`dtm` must be a non-empty 2D array/matrix')

    if idf_func is _idf:
        idf_opts = {}
        if 'smooth_log' in kwargs:
            idf_opts['smooth_log'] = kwargs.pop('smooth_log')
        if 'smooth_df' in kwargs:
            idf_opts['smooth_df'] = kwargs.pop('smooth_df')

        idf_vec = idf_func(dtm, **idf_opts)
    elif idf_func is idf_probabilistic and 'smooth' in kwargs:
        idf_vec = idf_func(dtm, smooth=kwargs.pop('smooth'))
    else:
        idf_vec = idf_func(dtm)

    tf_mat = tf_func(dtm, **kwargs)

    return tf_mat * idf_vec

def _conlltags2tree(
    sentence, chunk_types=("NP", "PP", "VP"), root_label="S", strict=False
):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, postag, chunktag) in sentence:
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word, postag))
        elif chunktag.startswith("B-"):
            tree.append(Tree(chunktag[2:], [(word, postag)]))
        elif chunktag.startswith("I-"):
            if (
                len(tree) == 0
                or not isinstance(tree[-1], Tree)
                or tree[-1].label() != chunktag[2:]
            ):
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word, postag)]))
            else:
                tree[-1].append((word, postag))
        elif chunktag == "O-":
            tree.append((word, postag))
        else:
            raise ValueError(f"Bad conll tag {chunktag!r}")
    return tree
  
class Tree(list):
    r"""
    A Tree represents a hierarchical grouping of leaves and subtrees.
    For example, each constituent in a syntax tree is represented by a single Tree.
    A tree's children are encoded as a list of leaves and subtrees,
    where a leaf is a basic (non-tree) value; and a subtree is a
    nested Tree.
        >>> from nltk.tree import Tree
        >>> print(Tree(1, [2, Tree(3, [4]), 5]))
        (1 2 (3 4) 5)
        >>> vp = Tree('VP', [Tree('V', ['saw']),
        ...                  Tree('NP', ['him'])])
        >>> s = Tree('S', [Tree('NP', ['I']), vp])
        >>> print(s)
        (S (NP I) (VP (V saw) (NP him)))
        >>> print(s[1])
        (VP (V saw) (NP him))
        >>> print(s[1,1])
        (NP him)
        >>> t = Tree.fromstring("(S (NP I) (VP (V saw) (NP him)))")
        >>> s == t
        True
        >>> t[1][1].set_label('X')
        >>> t[1][1].label()
        'X'
        >>> print(t)
        (S (NP I) (VP (V saw) (X him)))
        >>> t[0], t[1,1] = t[1,1], t[0]
        >>> print(t)
        (S (X him) (VP (V saw) (NP I)))
    The length of a tree is the number of children it has.
        >>> len(t)
        2
    The set_label() and label() methods allow individual constituents
    to be labeled.  For example, syntax trees use this label to specify
    phrase tags, such as "NP" and "VP".
    Several Tree methods use "tree positions" to specify
    children or descendants of a tree.  Tree positions are defined as
    follows:
      - The tree position *i* specifies a Tree's *i*\ th child.
      - The tree position ``()`` specifies the Tree itself.
      - If *p* is the tree position of descendant *d*, then
        *p+i* specifies the *i*\ th child of *d*.
    I.e., every tree position is either a single index *i*,
    specifying ``tree[i]``; or a sequence *i1, i2, ..., iN*,
    specifying ``tree[i1][i2]...[iN]``.
    Construct a new tree.  This constructor can be called in one
    of two ways:
    - ``Tree(label, children)`` constructs a new tree with the
        specified label and list of children.
    - ``Tree.fromstring(s)`` constructs a new tree by parsing the string ``s``.
    """

    def __init__(self, node, children=None):
        if children is None:
            raise TypeError(
                "%s: Expected a node value and child list " % type(self).__name__
            )
        elif isinstance(children, str):
            raise TypeError(
                "%s() argument 2 should be a list, not a "
                "string" % type(self).__name__
            )
        else:
            list.__init__(self, children)
            self._label = node

    # ////////////////////////////////////////////////////////////
    # Comparison operators
    # ////////////////////////////////////////////////////////////

    def __eq__(self, other):
        return self.__class__ is other.__class__ and (self._label, list(self)) == (
            other._label,
            list(other),
        )

    def __lt__(self, other):
        if not isinstance(other, Tree):
            # raise_unorderable_types("<", self, other)
            # Sometimes children can be pure strings,
            # so we need to be able to compare with non-trees:
            return self.__class__.__name__ < other.__class__.__name__
        elif self.__class__ is other.__class__:
            return (self._label, list(self)) < (other._label, list(other))
        else:
            return self.__class__.__name__ < other.__class__.__name__

    # @total_ordering doesn't work here, since the class inherits from a builtin class
    __ne__ = lambda self, other: not self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __le__ = lambda self, other: self < other or self == other
    __ge__ = lambda self, other: not self < other

    # ////////////////////////////////////////////////////////////
    # Disabled list operations
    # ////////////////////////////////////////////////////////////

    def __mul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __rmul__(self, v):
        raise TypeError("Tree does not support multiplication")

    def __add__(self, v):
        raise TypeError("Tree does not support addition")

    def __radd__(self, v):
        raise TypeError("Tree does not support addition")

    # ////////////////////////////////////////////////////////////
    # Indexing (with support for tree positions)
    # ////////////////////////////////////////////////////////////

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                return self
            elif len(index) == 1:
                return self[index[0]]
            else:
                return self[index[0]][index[1:]]
        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            return list.__setitem__(self, index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError("The tree position () may not be " "assigned to.")
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def __delitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__delitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError("The tree position () may not be deleted.")
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    # ////////////////////////////////////////////////////////////
    # Basic tree operations
    # ////////////////////////////////////////////////////////////
 
    def _get_node(self):
        """Outdated method to access the node value; use the label() method instead."""

    def _set_node(self, value):
        """Outdated method to set the node value; use the set_label() method instead."""

    node = property(_get_node, _set_node)

    def label(self):
        """
        Return the node label of the tree.
            >>> t = Tree.fromstring('(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))')
            >>> t.label()
            'S'
        :return: the node label (typically a string)
        :rtype: any
        """
        return self._label

    def set_label(self, label):
        """
        Set the node label of the tree.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.set_label("T")
            >>> print(t)
            (T (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))
        :param label: the node label (typically a string)
        :type label: any
        """
        self._label = label

    def leaves(self):
        """
        Return the leaves of the tree.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.leaves()
            ['the', 'dog', 'chased', 'the', 'cat']
        :return: a list containing this tree's leaves.
            The order reflects the order of the
            leaves in the tree's hierarchical structure.
        :rtype: list
        """
        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves())
            else:
                leaves.append(child)
        return leaves

    def flatten(self):
        """
        Return a flat version of the tree, with all non-root non-terminals removed.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> print(t.flatten())
            (S the dog chased the cat)
        :return: a tree consisting of this tree's root connected directly to
            its leaves, omitting all intervening non-terminal nodes.
        :rtype: Tree
        """
        return Tree(self.label(), self.leaves())

    def height(self):
        """
        Return the height of the tree.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.height()
            5
            >>> print(t[0,0])
            (D the)
            >>> t[0,0].height()
            2
        :return: The height of this tree.  The height of a tree
            containing no children is 1; the height of a tree
            containing only leaves is 2; and the height of any other
            tree is one plus the maximum of its children's
            heights.
        :rtype: int
        """
        max_child_height = 0
        for child in self:
            if isinstance(child, Tree):
                max_child_height = max(max_child_height, child.height())
            else:
                max_child_height = max(max_child_height, 1)
        return 1 + max_child_height

    def treepositions(self, order="preorder"):
        """
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.treepositions() # doctest: +ELLIPSIS
            [(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
            >>> for pos in t.treepositions('leaves'):
            ...     t[pos] = t[pos][::-1].upper()
            >>> print(t)
            (S (NP (D EHT) (N GOD)) (VP (V DESAHC) (NP (D EHT) (N TAC))))
        :param order: One of: ``preorder``, ``postorder``, ``bothorder``,
            ``leaves``.
        """
        positions = []
        if order in ("preorder", "bothorder"):
            positions.append(())
        for i, child in enumerate(self):
            if isinstance(child, Tree):
                childpos = child.treepositions(order)
                positions.extend((i,) + p for p in childpos)
            else:
                positions.append((i,))
        if order in ("postorder", "bothorder"):
            positions.append(())
        return positions

    def subtrees(self, filter=None):
        """
        Generate all the subtrees of this tree, optionally restricted
        to trees matching the filter function.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> for s in t.subtrees(lambda t: t.height() == 2):
            ...     print(s)
            (D the)
            (N dog)
            (V chased)
            (D the)
            (N cat)
        :type filter: function
        :param filter: the function to filter all local trees
        """
        if not filter or filter(self):
            yield self
        for child in self:
            if isinstance(child, Tree):
                yield from child.subtrees(filter)

    def productions(self):
        """
        Generate the productions that correspond to the non-terminal nodes of the tree.
        For each subtree of the form (P: C1 C2 ... Cn) this produces a production of the
        form P -> C1 C2 ... Cn.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.productions() # doctest: +NORMALIZE_WHITESPACE
            [S -> NP VP, NP -> D N, D -> 'the', N -> 'dog', VP -> V NP, V -> 'chased',
            NP -> D N, D -> 'the', N -> 'cat']
        :rtype: list(Production)
        """

        if not isinstance(self._label, str):
            raise TypeError(
                "Productions can only be generated from trees having node labels that are strings"
            )

        prods = [Production(Nonterminal(self._label), _child_names(self))]
        for child in self:
            if isinstance(child, Tree):
                prods += child.productions()
        return prods

    def pos(self):
        """
        Return a sequence of pos-tagged words extracted from the tree.
            >>> t = Tree.fromstring("(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")
            >>> t.pos()
            [('the', 'D'), ('dog', 'N'), ('chased', 'V'), ('the', 'D'), ('cat', 'N')]
        :return: a list of tuples containing leaves and pre-terminals (part-of-speech tags).
            The order reflects the order of the leaves in the tree's hierarchical structure.
        :rtype: list(tuple)
        """
        pos = []
        for child in self:
            if isinstance(child, Tree):
                pos.extend(child.pos())
            else:
                pos.append((child, self._label))
        return pos

    def leaf_treeposition(self, index):
        """
        :return: The tree position of the ``index``-th leaf in this
            tree.  I.e., if ``tp=self.leaf_treeposition(i)``, then
            ``self[tp]==self.leaves()[i]``.
        :raise IndexError: If this tree contains fewer than ``index+1``
            leaves, or if ``index<0``.
        """
        if index < 0:
            raise IndexError("index must be non-negative")

        stack = [(self, ())]
        while stack:
            value, treepos = stack.pop()
            if not isinstance(value, Tree):
                if index == 0:
                    return treepos
                else:
                    index -= 1
            else:
                for i in range(len(value) - 1, -1, -1):
                    stack.append((value[i], treepos + (i,)))

        raise IndexError("index must be less than or equal to len(self)")

    def treeposition_spanning_leaves(self, start, end):
        """
        :return: The tree position of the lowest descendant of this
            tree that dominates ``self.leaves()[start:end]``.
        :raise ValueError: if ``end <= start``
        """
        if end <= start:
            raise ValueError("end must be greater than start")
        # Find the tree positions of the start & end leaves, and
        # take the longest common subsequence.
        start_treepos = self.leaf_treeposition(start)
        end_treepos = self.leaf_treeposition(end - 1)
        # Find the first index where they mismatch:
        for i in range(len(start_treepos)):
            if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
                return start_treepos[:i]
        return start_treepos

    # ////////////////////////////////////////////////////////////
    # Transforms
    # ////////////////////////////////////////////////////////////

    def chomsky_normal_form(
        self,
        factor="right",
        horzMarkov=None,
        vertMarkov=0,
        childChar="|",
        parentChar="^",
    ):
        """
        This method can modify a tree in three ways:
          1. Convert a tree into its Chomsky Normal Form (CNF)
             equivalent -- Every subtree has either two non-terminals
             or one terminal as its children.  This process requires
             the creation of more"artificial" non-terminal nodes.
          2. Markov (vertical) smoothing of children in new artificial
             nodes
          3. Horizontal (parent) annotation of nodes
        :param factor: Right or left factoring method (default = "right")
        :type  factor: str = [left|right]
        :param horzMarkov: Markov order for sibling smoothing in artificial nodes (None (default) = include all siblings)
        :type  horzMarkov: int | None
        :param vertMarkov: Markov order for parent smoothing (0 (default) = no vertical annotation)
        :type  vertMarkov: int | None
        :param childChar: A string used in construction of the artificial nodes, separating the head of the
                          original subtree from the child nodes that have yet to be expanded (default = "|")
        :type  childChar: str
        :param parentChar: A string used to separate the node representation from its vertical annotation
        :type  parentChar: str
        """
        from nltk.tree.transforms import chomsky_normal_form

        chomsky_normal_form(self, factor, horzMarkov, vertMarkov, childChar, parentChar)

    def un_chomsky_normal_form(
        self, expandUnary=True, childChar="|", parentChar="^", unaryChar="+"
    ):
        """
        This method modifies the tree in three ways:
          1. Transforms a tree in Chomsky Normal Form back to its
             original structure (branching greater than two)
          2. Removes any parent annotation (if it exists)
          3. (optional) expands unary subtrees (if previously
             collapsed with collapseUnary(...) )
        :param expandUnary: Flag to expand unary or not (default = True)
        :type  expandUnary: bool
        :param childChar: A string separating the head node from its children in an artificial node (default = "|")
        :type  childChar: str
        :param parentChar: A string separating the node label from its parent annotation (default = "^")
        :type  parentChar: str
        :param unaryChar: A string joining two non-terminals in a unary production (default = "+")
        :type  unaryChar: str
        """
        from nltk.tree.transforms import un_chomsky_normal_form

        un_chomsky_normal_form(self, expandUnary, childChar, parentChar, unaryChar)

    def collapse_unary(self, collapsePOS=False, collapseRoot=False, joinChar="+"):
        """
        Collapse subtrees with a single child (ie. unary productions)
        into a new non-terminal (Tree node) joined by 'joinChar'.
        This is useful when working with algorithms that do not allow
        unary productions, and completely removing the unary productions
        would require loss of useful information.  The Tree is modified
        directly (since it is passed by reference) and no value is returned.
        :param collapsePOS: 'False' (default) will not collapse the parent of leaf nodes (ie.
                            Part-of-Speech tags) since they are always unary productions
        :type  collapsePOS: bool
        :param collapseRoot: 'False' (default) will not modify the root production
                             if it is unary.  For the Penn WSJ treebank corpus, this corresponds
                             to the TOP -> productions.
        :type collapseRoot: bool
        :param joinChar: A string used to connect collapsed node values (default = "+")
        :type  joinChar: str
        """
        from nltk.tree.transforms import collapse_unary

        collapse_unary(self, collapsePOS, collapseRoot, joinChar)

    # ////////////////////////////////////////////////////////////
    # Convert, copy
    # ////////////////////////////////////////////////////////////

    @classmethod
    def convert(cls, tree):
        """
        Convert a tree between different subtypes of Tree.  ``cls`` determines
        which class will be used to encode the new tree.
        :type tree: Tree
        :param tree: The tree that should be converted.
        :return: The new Tree.
        """
        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            return cls(tree._label, children)
        else:
            return tree

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy(deep=True)

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self)
        else:
            return type(self).convert(self)

    def _frozen_class(self):
        from nltk.tree.immutable import ImmutableTree

        return ImmutableTree

    def freeze(self, leaf_freezer=None):
        frozen_class = self._frozen_class()
        if leaf_freezer is None:
            newcopy = frozen_class.convert(self)
        else:
            newcopy = self.copy(deep=True)
            for pos in newcopy.treepositions("leaves"):
                newcopy[pos] = leaf_freezer(newcopy[pos])
            newcopy = frozen_class.convert(newcopy)
        hash(newcopy)  # Make sure the leaves are hashable.
        return newcopy

    # ////////////////////////////////////////////////////////////
    # Parsing
    # ////////////////////////////////////////////////////////////

    @classmethod
    def fromstring(
        cls,
        s,
        brackets="()",
        read_node=None,
        read_leaf=None,
        node_pattern=None,
        leaf_pattern=None,
        remove_empty_top_bracketing=False,
    ):
        """
        Read a bracketed tree string and return the resulting tree.
        Trees are represented as nested brackettings, such as::
          (S (NP (NNP John)) (VP (V runs)))
        :type s: str
        :param s: The string to read
        :type brackets: str (length=2)
        :param brackets: The bracket characters used to mark the
            beginning and end of trees and subtrees.
        :type read_node: function
        :type read_leaf: function
        :param read_node, read_leaf: If specified, these functions
            are applied to the substrings of ``s`` corresponding to
            nodes and leaves (respectively) to obtain the values for
            those nodes and leaves.  They should have the following
            signature:
               read_node(str) -> value
            For example, these functions could be used to process nodes
            and leaves whose values should be some type other than
            string (such as ``FeatStruct``).
            Note that by default, node strings and leaf strings are
            delimited by whitespace and brackets; to override this
            default, use the ``node_pattern`` and ``leaf_pattern``
            arguments.
        :type node_pattern: str
        :type leaf_pattern: str
        :param node_pattern, leaf_pattern: Regular expression patterns
            used to find node and leaf substrings in ``s``.  By
            default, both nodes patterns are defined to match any
            sequence of non-whitespace non-bracket characters.
        :type remove_empty_top_bracketing: bool
        :param remove_empty_top_bracketing: If the resulting tree has
            an empty node label, and is length one, then return its
            single child instead.  This is useful for treebank trees,
            which sometimes contain an extra level of bracketing.
        :return: A tree corresponding to the string representation ``s``.
            If this class method is called using a subclass of Tree,
            then it will return a tree of that type.
        :rtype: Tree
        """
        if not isinstance(brackets, str) or len(brackets) != 2:
            raise TypeError("brackets must be a length-2 string")
        if re.search(r"\s", brackets):
            raise TypeError("whitespace brackets not allowed")
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = rf"[^\s{open_pattern}{close_pattern}]+"
        if leaf_pattern is None:
            leaf_pattern = rf"[^\s{open_pattern}{close_pattern}]+"
        token_re = re.compile(
            r"%s\s*(%s)?|%s|(%s)"
            % (open_pattern, node_pattern, close_pattern, leaf_pattern)
        )
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])]  # list of (node, children) tuples
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(s, match, "end-of-string")
                label = token[1:].lstrip()
                if read_node is not None:
                    label = read_node(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(s, match, open_b)
                    else:
                        cls._parse_error(s, match, "end-of-string")
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(s, match, open_b)
                if read_leaf is not None:
                    token = read_leaf(token)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, "end-of-string", close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, "end-of-string", open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == "" and len(tree) == 1:
            tree = tree[0]
        # return the tree.
        return tree

    @classmethod
    def _parse_error(cls, s, match, expecting):
        """
        Display a friendly error message when parsing a tree string fails.
        :param s: The string we're parsing.
        :param match: regexp match of the problem token.
        :param expecting: what we expected to see instead.
        """
        # Construct a basic error message
        if match == "end-of-string":
            pos, token = len(s), "end-of-string"
        else:
            pos, token = match.start(), match.group()
        msg = "%s.read(): expected %r but got %r\n%sat index %d." % (
            cls.__name__,
            expecting,
            token,
            " " * 12,
            pos,
        )
        # Add a display showing the error token itsels:
        s = s.replace("\n", " ").replace("\t", " ")
        offset = pos
        if len(s) > pos + 10:
            s = s[: pos + 10] + "..."
        if pos > 10:
            s = "..." + s[pos - 10 :]
            offset = 13
        msg += '\n{}"{}"\n{}^'.format(" " * 16, s, " " * (17 + offset))
        raise ValueError(msg)

    @classmethod
    def fromlist(cls, l):
        """
        :type l: list
        :param l: a tree represented as nested lists
        :return: A tree corresponding to the list representation ``l``.
        :rtype: Tree
        Convert nested lists to a NLTK Tree
        """
        if type(l) == list and len(l) > 0:
            label = repr(l[0])
            if len(l) > 1:
                return Tree(label, [cls.fromlist(child) for child in l[1:]])
            else:
                return label

    # ////////////////////////////////////////////////////////////
    # Visualization & String Representation
    # ////////////////////////////////////////////////////////////

    def draw(self):
        """
        Open a new window containing a graphical diagram of this tree.
        """
        from nltk.draw.tree import draw_trees

        draw_trees(self)

    def pretty_print(self, sentence=None, highlight=(), stream=None, **kwargs):
        """
        Pretty-print this tree as ASCII or Unicode art.
        For explanation of the arguments, see the documentation for
        `nltk.tree.prettyprinter.TreePrettyPrinter`.
        """
        from nltk.tree.prettyprinter import TreePrettyPrinter

        print(TreePrettyPrinter(self, sentence, highlight).text(**kwargs), file=stream)

    def __repr__(self):
        childstr = ", ".join(repr(c) for c in self)
        return "{}({}, [{}])".format(
            type(self).__name__,
            repr(self._label),
            childstr,
        )

    def _repr_svg_(self):
        from svgling import draw_tree

        return draw_tree(self)._repr_svg_()

    def __str__(self):
        return self.pformat()

    def pprint(self, **kwargs):
        """
        Print a string representation of this Tree to 'stream'
        """

        if "stream" in kwargs:
            stream = kwargs["stream"]
            del kwargs["stream"]
        else:
            stream = None
        print(self.pformat(**kwargs), file=stream)

    def pformat(self, margin=70, indent=0, nodesep="", parens="()", quotes=False):
        """
        :return: A pretty-printed string representation of this tree.
        :rtype: str
        :param margin: The right margin at which to do line-wrapping.
        :type margin: int
        :param indent: The indentation level at which printing
            begins.  This number is used to decide how far to indent
            subsequent lines.
        :type indent: int
        :param nodesep: A string that is used to separate the node
            from the children.  E.g., the default value ``':'`` gives
            trees like ``(S: (NP: I) (VP: (V: saw) (NP: it)))``.
        """

        # Try writing it on one line.
        s = self._pformat_flat(nodesep, parens, quotes)
        if len(s) + indent < margin:
            return s

        # If it doesn't fit on one line, then write it on multi-lines.
        if isinstance(self._label, str):
            s = f"{parens[0]}{self._label}{nodesep}"
        else:
            s = f"{parens[0]}{repr(self._label)}{nodesep}"
        for child in self:
            if isinstance(child, Tree):
                s += (
                    "\n"
                    + " " * (indent + 2)
                    + child.pformat(margin, indent + 2, nodesep, parens, quotes)
                )
            elif isinstance(child, tuple):
                s += "\n" + " " * (indent + 2) + "/".join(child)
            elif isinstance(child, str) and not quotes:
                s += "\n" + " " * (indent + 2) + "%s" % child
            else:
                s += "\n" + " " * (indent + 2) + repr(child)
        return s + parens[1]

    def pformat_latex_qtree(self):
        r"""
        Returns a representation of the tree compatible with the
        LaTeX qtree package. This consists of the string ``\Tree``
        followed by the tree represented in bracketed notation.
        For example, the following result was generated from a parse tree of
        the sentence ``The announcement astounded us``::
          \Tree [.I'' [.N'' [.D The ] [.N' [.N announcement ] ] ]
              [.I' [.V'' [.V' [.V astounded ] [.N'' [.N' [.N us ] ] ] ] ] ] ]
        See https://www.ling.upenn.edu/advice/latex.html for the LaTeX
        style file for the qtree package.
        :return: A latex qtree representation of this tree.
        :rtype: str
        """
        reserved_chars = re.compile(r"([#\$%&~_\{\}])")

        pformat = self.pformat(indent=6, nodesep="", parens=("[.", " ]"))
        return r"\Tree " + re.sub(reserved_chars, r"\\\1", pformat)

    def _pformat_flat(self, nodesep, parens, quotes):
        childstrs = []
        for child in self:
            if isinstance(child, Tree):
                childstrs.append(child._pformat_flat(nodesep, parens, quotes))
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, str) and not quotes:
                childstrs.append("%s" % child)
            else:
                childstrs.append(repr(child))
        if isinstance(self._label, str):
            return "{}{}{} {}{}".format(
                parens[0],
                self._label,
                nodesep,
                " ".join(childstrs),
                parens[1],
            )
        else:
            return "{}{}{} {}{}".format(
                parens[0],
                repr(self._label),
                nodesep,
                " ".join(childstrs),
                parens[1],
            )


def _child_names(tree):
    names = []
    for child in tree:
        if isinstance(child, Tree):
            names.append(Nonterminal(child._label))
        else:
            names.append(child)
    return names

def _conlltags2tree(
    sentence, chunk_types=("NP", "PP", "VP"), root_label="S", strict=False
):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, postag, chunktag) in sentence:
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word, postag))
        elif chunktag.startswith("B-"):
            tree.append(Tree(chunktag[2:], [(word, postag)]))
        elif chunktag.startswith("I-"):
            if (
                len(tree) == 0
                or not isinstance(tree[-1], Tree)
                or tree[-1].label() != chunktag[2:]
            ):
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word, postag)]))
            else:
                tree[-1].append((word, postag))
        elif chunktag == "O-":
            tree.append((word, postag))
        else:
            raise ValueError(f"Bad conll tag {chunktag!r}")
    return tree
  
