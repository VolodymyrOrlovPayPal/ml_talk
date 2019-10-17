#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:python

echo 'Naive implementation:'
python -m timeit "from cholesky import test_naive_cholesky; test_naive_cholesky()"

echo 'Numpy implementation:'
python -m timeit "from cholesky import test_numpy_cholesky; test_numpy_cholesky()"
