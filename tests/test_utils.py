import numpy as np
import scipy.sparse
import pytest

from lccd_python import utils


TEST_TIMES = 50


def test_insert_binary_col_binary_csc():
    dtypes = [np.float32, np.uint8]
    for _ in range(TEST_TIMES):
        for dtype in dtypes:
            rows, cols = np.random.randint(30, 50, 2)
            for i in [0, np.random.randint(1, cols)]:
                mat = np.random.rand(rows, cols)
                mat[mat > 0] = 1
                mat[mat <= 0] = 0
                mat = mat.astype(dtype)
                mat = scipy.sparse.csc_matrix(mat)

                num_nonzero = np.random.randint(0, rows+1)
                indices = np.sort(np.random.choice(range(rows), num_nonzero, replace=False))

                mat_dense = mat.toarray()
                dense_values = np.zeros(rows, dtype=dtype)
                dense_values[indices] = 1
                mat_dense_inserted = np.insert(mat_dense, i, dense_values, axis=1)
                utils.insert_binary_col_binary_csc(mat, indices, i)
                np.testing.assert_array_equal(mat_dense_inserted, mat.toarray())


def test_delete_row_csr():
    dtypes = [np.float32, np.uint8]
    for _ in range(TEST_TIMES):
        for dtype in dtypes:
            rows, cols = np.random.randint(30, 50, 2)
            mat = scipy.sparse.random(rows, cols, density=0.02, format='csr', dtype=dtype)
            idx = np.random.randint(0, rows)
            correct = np.delete(mat.toarray(), idx, 0)
            res = utils.delete_row_csr(mat, idx)
            np.testing.assert_array_equal(correct, res.toarray())

def test_delete_col_csc():
    dtypes = [np.float32, np.uint8]
    for _ in range(TEST_TIMES):
        for dtype in dtypes:
            rows, cols = np.random.randint(10, 20, 2)
            mat = scipy.sparse.random(rows, cols, density=0.01, format='csc', dtype=dtype)
            idx = np.random.randint(0, cols)
            correct = np.delete(mat.toarray(), idx, 1)
            res = utils.delete_col_csc(mat, idx)
            np.testing.assert_array_equal(correct, res.toarray())


def test_delete_col_csc_inplace():
    dtypes = [np.float32, np.uint8]
    for _ in range(TEST_TIMES):
        for dtype in dtypes:
            rows, cols = np.random.randint(10, 20, 2)
            mat = scipy.sparse.random(rows, cols, density=0.01, format='csc', dtype=dtype)
            idx = np.random.randint(0, cols)
            correct = np.delete(mat.toarray(), idx, 1)
            utils.delete_col_csc_inplace(mat, idx)
            res = mat
            np.testing.assert_array_equal(correct, res.toarray())


def test_count_nonzero_values_in_col_csc():
    dtypes = [np.float32, np.uint8]
    for _ in range(TEST_TIMES):
        for dtype in dtypes:
            rows, cols = np.random.randint(10, 20, 2)
            mat = scipy.sparse.random(rows, cols, density=0.01, format='csc', dtype=dtype)
            idx = np.random.randint(0, cols)
            res = utils.count_nonzero_values_in_col_csc(mat, idx)
            np.sum(mat[:, idx]!=0) == mat[:, idx].count_nonzero() == np.count_nonzero(mat[:, idx].toarray()) == res


def test_intersect_unique_sorted_1d():
    for _ in range(TEST_TIMES):
        arr1 = np.sort(np.unique(np.random.randint(0, 100, 50)))
        arr2 = np.sort(np.unique(np.random.randint(0, 100, 50)))
        correct = np.intersect1d(arr1, arr2)
        np.testing.assert_array_equal(correct, utils.intersect_unique_sorted_1d(arr1, arr2))


def test_array_to_lil_row():
    for _ in range(TEST_TIMES):
        n_rows, n_cols = np.random.randint(1, 100, 2)
        for arr in [np.random.rand(n_rows, n_cols), np.random.randint(0, 2, size=(n_rows, n_cols), dtype=np.uint8)]:
            rows, data = [], []
            for row_ in arr:
                datum, row = utils.array_to_lil_row(row_)
                rows.append(row)
                data.append(datum)
            dtype = row_.dtype
            lil_arr = scipy.sparse.lil_matrix((n_rows, n_cols), dtype=dtype)
            lil_arr.data = np.array(data, dtype='object')
            lil_arr.rows = np.array(rows, dtype='object')
            np.testing.assert_array_equal(arr, lil_arr.toarray())
