import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.3.3'
PACKAGE_NAME = 'docuscospacy'
AUTHOR = 'David Brown'
AUTHOR_EMAIL = 'dwb2@andrew.cmu.edu'
URL = 'https://github.com/browndw/docuscospacy'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Support for spaCy models trained on \
      DocuScope and the CLAWS7 tagset'
LONG_DESCRIPTION = (HERE / "README.rst").read_text()
LONG_DESC_TYPE = "text/x-rst"

INSTALL_REQUIRES = [
      'importlib-resources>=6.5.0',
      'polars>=1.30.0',
      'spacy>=3.8.0',
      'scipy>=1.15.0'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
