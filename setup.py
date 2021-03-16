import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'circular-coordinates'
AUTHOR = 'Mikael Vejdemo Johansson, Bilal AbdulRahman'
AUTHOR_EMAIL = 'mikael.vejdemojohansson@csi.cuny.edu, bilal.z.work@gmail.com'
URL = 'https://github.com/appliedtopology/circular-coordinates'

LICENSE = 'MIT'
DESCRIPTION = 'Package contains tools used to create and plot circular coordinates from persistent cohomology '
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'scipy',
      'sklearn',
      'ripser'
]

keywords=['python', 'cohomology','circular coordinates']

classifiers= [
            "Programming Language :: Python :: 3",
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
      packages=find_packages(),
      keywords=keywords,
      classifiers=classifiers
      )

