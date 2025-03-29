from setuptools import setup, find_packages
import sys

if sys.version_info >= (3, 12):
    raise RuntimeError("⚠️ golem is not compatible with Python 3.12+. Please use Python 3.11 or below.")

setup(
    name='golem',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8, <=3.11',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'opencv-python',
        'pylops',
        'numba'
    ],
    author='Ammir Ayman Karsou',
    description='Seismic processing toolkit',
)