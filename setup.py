from setuptools import setup
from setuptools import find_packages

setup(name='Neural Network',
      version='0.1.0',
      description='Deep Learning with Keras',
      author='',
      author_email='',
      url='',
      download_url='',
      license='',
      install_requires=[
          'numpy>=1.9.1',
          'tensorflow>=1.6.0',
          'protobuf>=3.5.1',
          'h5py>=2.7.1',
          'pandas>=0.22.0'
      ],
      extras_require={
          'tests': [
              'pytest',
              'pytest-pep8',
              'pytest-xdist',
              'pytest-cov',
              'pandas',
              'requests'
          ],
      },
      classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6'
      ],
      packages=find_packages())
