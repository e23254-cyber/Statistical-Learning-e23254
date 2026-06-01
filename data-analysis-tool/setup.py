from setuptools import setup, find_packages

setup(
    name='data-analysis-tool',
    version='0.1',
    packages=find_packages(),
    install_requires=[] # Left empty so Colab uses its own built-in versions
)
