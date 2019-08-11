import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multi-imbalance",
    version="0.0.1",
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
)