''' 
    to enable calling sibiling packages? 
    ref: https://stackoverflow.com/questions/6323860/sibling-package-imports 
'''
from setuptools import setup, find_packages
setup(name='audmem_models', version='1.0', packages=find_packages())