from setuptools import setup, find_packages

setup(
    name="causal-pipe",
    version="0.3.0",
    author="Buchard, Albert",
    author_email="albert.buchard@gmail.com",
    description="A Python package streamlining the causal discovery pipeline for easy use.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/albertbuchard/causal-pipe",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "scikit-learn>=0.22.0",
        "causal-learn==0.1.3.8",
        "bcsl-python==0.8.0",
        "rpy2==3.5.16",
        "npeet-plus==0.2.0",
        "networkx==3.2.1",
        "pandas==2.2.3",
        "factor_analyzer==0.5.1",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "graphviz==0.20.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)