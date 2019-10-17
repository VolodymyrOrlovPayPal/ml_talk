#/usr/bin/env python

import sys
import math
import numpy as np


def naive_cholesky(m):
    n_cols, n_rows = m.shape
    assert n_cols == n_rows, 'Can do Cholesky decomposition only for square matrix'
    for j in range(0, n_rows):
        d = .0
        for k in range(0, j):
            s = .0
            for i in range(0, k):
                s += m[k, i] * m[j, i]
            s = (m[j, k] - s) / m[k, k]
            m[j, k] = s
            d = d + s * s
        d = m[j, j] - d

        assert d > 0, 'Matrix is not positive definite'

        m[j, j] = math.sqrt(d)

    return m


def compare(left, right):
    assert left.shape == right.shape, 'left.shape should == right.shape'
    for r in range(0, left.shape[0]):
        for c in range(0, left.shape[1]):
            if r >= c:
                assert np.abs(left[r, c] - right[r, c]) < 10**-6, 'left != right, left({}, {}) != right({}, {})'.format(r, c, r, c)


def generate_positive_definite_m(*dn):
    m = np.random.rand(*dn)
    return np.dot(m, m.transpose())


def test_naive_cholesky():
    m = generate_positive_definite_m(16, 16)
    naive_cholesky(m)


def test_numpy_cholesky():
    m = generate_positive_definite_m(16, 16)
    np.linalg.cholesky(m)


def main():
    m = generate_positive_definite_m(3, 3)
    c = np.linalg.cholesky(m)
    naive_cholesky(m)
    compare(c, m)


if __name__ == "__main__":
    main()
