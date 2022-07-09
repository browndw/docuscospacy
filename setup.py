import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'docuscospacy'
AUTHOR = 'David Brown'
AUTHOR_EMAIL = 'dwb2@andrew.cmu.edu'
URL = 'https://github.com/browndw/docuscospacy'

LICENSE = 'Apache License 2.0'
DESCRIPTION = 'Describe your package in one sentence'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy>=1.22.0',
      'pandas>=1.4.0',
      'tmtoolkit>=0.11.0',
      're',
      'string'
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
