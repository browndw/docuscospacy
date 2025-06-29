import pathlib
import re
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

with open("docuscospacy/__init__.py") as f:
    init = f.read()
    VERSION = re.search(r'__version__ = ["\']([^"\']+)["\']', init).group(1)
    AUTHOR = re.search(r'__author__ = ["\']([^"\']+)["\']', init).group(1)
    AUTHOR_EMAIL = re.search(r'__email__ = ["\']([^"\']+)["\']', init).group(1)

PACKAGE_NAME = "docuscospacy"
URL = "https://github.com/browndw/docuscospacy"

LICENSE = "Apache License 2.0"
DESCRIPTION = "Support for spaCy models trained on \
      DocuScope and the CLAWS7 tagset"
LONG_DESCRIPTION = (HERE / "README.rst").read_text()
LONG_DESC_TYPE = "text/x-rst"

INSTALL_REQUIRES = [
    "importlib-resources>=6.5.0",
    "polars>=1.30.0",
    "spacy>=3.8.0",
    "scipy>=1.10.0",
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data={"docuscospacy": ["data/*.parquet"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
)
