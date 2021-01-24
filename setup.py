from setuptools import setup, find_packages

setup(name='tfmonopoles',
    version='1.0',
    author='David Ho',
    install_requires=[
        'numpy',
        'tensorflow>=2.2.0'
    ],
    packages=find_packages())