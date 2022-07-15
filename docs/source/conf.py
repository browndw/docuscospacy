# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import date

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'docuscospacy'
copyright = f'{date.today().year}, David Brown'
author = 'David Brown'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex',
    'sphinx_rtd_theme'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Add path for bib files
bibtex_bibfiles = ['refs.bib']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ['**.ipynb_checkpoints']

bibtex_bibliography_header = ".. rubric:: References"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# type hints
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
