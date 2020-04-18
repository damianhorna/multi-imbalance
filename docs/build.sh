#!/usr/bin/env bash
rm -Rf docs/build docs/source/docstring
sphinx-apidoc -f -o docs/source/docstring/ multi_imbalance --append-syspath
cd docs && make html