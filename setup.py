import setuptools # pragma no cover

with open("README.md", "r") as fh:  # pragma no cover

    long_description = fh.read()

setuptools.setup(  # pragma no cover
    name="multi-imbalance",
    version="0.0.12",
    author="Damian Horna, Kamil Pluciński, Hanna Klimczak, Jacek Grycza",
    author_email="horna.damian@gmail.com",
    description="Python package for tackling multiclass imbalance problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/damian-horna/multi-imbalance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=[
        "numpy>=1.17.0",
        "scikit-learn>=0.21.3",
        "pandas>=0.25.1",
        "pytest>=5.1.2",
        "imbalanced-learn>=0.6.1",
        "coverage>=5.1",
        "pytest-cov>=2.8.1",
        "IPython>=7.13.0",
        "seaborn>=0.10.1",
        "matplotlib>=3.2.1",
    ]
)
