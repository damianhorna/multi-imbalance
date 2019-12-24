import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi-imbalance",
    version="0.0.3",
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
    ],
    install_requires=[
        "numpy>=1.17.0",
        "scikit-learn>=0.21.3",
        "pandas>=0.25.1",
        "seaborn>=0.9.0",
        "pytest>=5.1.2",
        "sklearn>=0.0",
        "imbalanced-learn",
    ]
)
