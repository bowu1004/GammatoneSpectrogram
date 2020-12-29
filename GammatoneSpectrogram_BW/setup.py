"""
    You can install directly from this git repository using:
        pip install git+https://github.com/bowu1004/GammatoneSpectrogram.git

    ...or you can clone the git repository however you prefer, and do:
        pip install .

    ...or:
        python setup.py install
    from the cloned tree.

"""
from setuptools import setup, find_packages

setup(
    name = "GammatoneSpectrogram_BW",
    version = "0.1",
    packages = find_packages(),

    install_requires = [
        'numpy',
        'scipy',
        'nose',
        'mock',
        'librosa',
        'matplotlib',
    ],

    entry_points = {
        'console_scripts': [
            'gammatone = GammatoneSpectrogram.main_demo_BW:main',
        ]
    }
)
