|docuscope|

docuscospacy: Support for spaCy models trained on DocuScope and the CLAWS7 tagset
=================================================================================
|pypi| |pypi_downloads| |rtd| |zenodo|

The *docuscospacy* package contains a set of functions to facilitate the processing of tagged corpora using:

* `en_docusco_spacy <https://huggingface.co/browndw/en_docusco_spacy>`_ -- a spaCy model trained on the `CLAWS7 <https://ucrel.lancs.ac.uk/claws7tags.html>`_ tagset and `DocuScope <https://docuscospacy.readthedocs.io/en/latest/docuscope.html>`_

The current version of the package runs in `polars <https://docs.pola.rs/api/python/stable/reference/index.html>`_

The package can also convert a corpus to and from:

* `tmtoolkit <https://tmtoolkit.readthedocs.io/en/latest/>`_ --  a set of tools for text mining and topic modeling

The documentation for docuscospacy is available on `docuscospacy.readthedocs.org <https://docuscospacy.readthedocs.org>`_ and the GitHub code repository is on
`github.com/browndw/docuscospacy <https://github.com/browndw/docuscospacy>`_.

Requirements and installation
-----------------------------

**docuscospacy works with Python 3.8 or newer (tested up to Python 3.10).** It also requires `spacy >= 3.3 <https://spacy.io/usage>`_.

The recommended way of installing *docuscospacy* is to:

- create and activate a `Python Virtual Environment ("venv") <https://docs.python.org/3/tutorial/venv.html>`_ 
- install `spacy <https://spacy.io/usage>`_
- install `docuscospacy <https://docuscospacy.readthedocs.io/en/latest/install.html>`_
- download the `en_docusco_spacy <https://huggingface.co/browndw/en_docusco_spacy>`_ model

.. code-block:: text

    pip install docuscospacy

Note that installing the model depends on your spaCy version. Some versions allow:

.. code-block:: model-1

    !pip install https://huggingface.co/browndw/en_docusco_spacy/resolve/main/en_docusco_spacy-any-py3-none-any.whl

But new ones may require:

.. code-block:: model-2

    pip install "en_docusco_spacy @ https://huggingface.co/browndw/en_docusco_spacy/resolve/main/en_docusco_spacy-any-py3-none-any.whl"

Features
--------

Corpus analysis
^^^^^^^^^^^^^^^

The docuscospacy package supports the post-tagging generation of:

- `Tagged token frequency tables <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Frequency-tables>`_
- `Tag frequency tables <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Tags-tables>`_
- `Ngram/ntag tables <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Ngram-tables>`_
- `Collocation tables around a node word and tag <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Collocations>`_
- `Document term matrices for tags <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Document-term-matrices-for-tags>`_
- `Keyword comparisons against a reference corpus <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#Keyword-tables>`_

**Outputs can be controlled either by part-of-speech or by DocuScope tag**. Thus, *can* as noun and *can* as verb, for example, can be disambiguated.

Additionally, tagged multi-token sequences are aggregated for analysis. So, for example, where *in spite of* is tagged as a token sequence, it is combined into a single token.

Other features
^^^^^^^^^^^^^^

- `KWIC tables <https://docuscospacy.readthedocs.io/en/latest/corpus_analysis.html#KWIC-tables>`_ that locate a node word in a center column with context columns on either side

Limits
------

* the model that this package is designed for has only been trained on English
* all data must reside in memory, i.e. no streaming of large data from the hard disk (which for example
  `Gensim <https://radimrehurek.com/gensim/>`_ supports)


License
-------

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/browndw/docuscospacy/blob/master/LICENSE>`_ file.

.. |docuscope| image:: https://avatars.githubusercontent.com/u/21162269?s=200&v=4
    :target: https://www.cmu.edu/dietrich/english/research-and-publications/docuscope.html
    :alt: DocuScope

.. |pypi| image:: https://badge.fury.io/py/docuscospacy.svg
    :target: https://badge.fury.io/py/docuscospacy
    :alt: PyPI Version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/docuscospacy
    :target: https://pypi.org/project/docuscospacy/
    :alt: Downloads from PyPI
        
.. |rtd| image:: https://readthedocs.org/projects/docuscospacy/badge/?version=latest
    :target: https://docuscospacy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |zenodo| image:: https://zenodo.org/badge/512227318.svg
    :target: https://zenodo.org/badge/latestdoi/512227318
    :alt: Citable Zenodo DOI