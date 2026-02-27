import os
from setuptools import setup


# Helper function to load up readme as long description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name = 'sspspace',
        version = '0.2',
        author='Nicole S.Y. Dumont, P. Michael Furlong', 
        author_email='ns2dumont@uwaterloo.ca, michael.furlong@uwaterloo.ca',
        description=('Algebraic Implementation of Spatial Semantic Pointers'),
        license = 'MIT',
        keywords = 'spatial semantic pointers, vector symbolic architecture',
        url='http://github.com/ctn-waterloo/sspspace',
        packages=['sspspace'],
        long_description=read('README.md'),
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3',
            'Environment :: Console', 
            'Topic :: Scientific/Engineering :: Artificial Intelligence'
            ],
        install_requires=[
            'numpy>=1.21.2',
            'scipy',
            'pytest',
            'mypy',
            'typing-extensions',
        ]
)
