"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup

install_requires = [
        'rdkit',
        'logging',
        'scipy',
]

setup(
    name = 'util',
    version='0.0.1',
    author='MaxWang',
    author_email='wangjy108@outlook.com',
    description=('UNIPF util part'),
    url='https://github.com/wangjy108/Uni-PF/tree/main/util',
    packages=['util'],
)
