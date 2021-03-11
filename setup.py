# https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
from setuptools import setup, find_packages

setup(
  name = 'pandas2numpy',
  version = '1.0',
  license='apache-2.0',
  description = 'Dataframe to tensor converter for deep learning.',
  author = 'NestorDemeure',
  #  author_email = 'your.email@domain.com',
  url = 'https://github.com/nestordemeure/pandas2numpy',
  keywords = ['pandas', 'numpy', 'tabular-data', 'deep-learning'],
  install_requires=[
          'numpy',
          'pandas'
      ],
  classifiers=[ # https://pypi.org/classifiers/
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3 :: Only',
  ],
  packages=find_packages(),
)
