from setuptools import setup, find_packages

setup(
    name='sta_cl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'einops'
    ],
)
