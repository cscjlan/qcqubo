import numpy as np
import time


class Sparse:
    def __init__(self, arr: np.ndarray):
        assert len(arr.shape) == 2
        assert arr.shape[0] == arr.shape[1]

        self.arr = arr
        nonzero_inds = np.where(self.arr != 0.0)
        values = self.arr[nonzero_inds]
        rows = [[] for _ in range(self.arr.shape[0])]
        for i, row in enumerate(nonzero_inds[0]):
            rows[row].append((nonzero_inds[1][i], values[i]))

        print(len(rows[1]))
        print(len(np.where(nonzero_inds[0] == 1.0)[0]))


def solve(arr: np.ndarray):
    e = 100000
    dv = 0.000336499
    nv = arr.shape[0]

    si = np.ones(nv)

    co = np.ones(nv)
    co = 2 * dv * co

    it = 100000  # number of iterations, as large as possible, millions...
    for _ in range(it):
        s = np.random.randint(2, size=nv)
        sum_s = 2 * dv * np.sum(s)
        es = np.dot(s, np.dot(arr, s)) + sum_s
        if es < e:
            e = es
            si = s.copy()

    print(e)
    print(si)
    # validate = np.loadtxt("validate.csv", delimiter=",")
    # assert (validate == si).all()


def sparse_solve(arr: np.ndarray):
    sparse = Sparse(arr)
    print(sparse.arr.shape)


def main():
    np.random.seed(6)
    sqv = np.genfromtxt("sqv.csv", delimiter=",")

    solve(sqv)
    # sparse_solve(sqv)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"time taken: {time.time() - t}")
