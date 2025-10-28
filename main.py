import numpy as np
import time


def main():
    np.random.seed(6)

    nv = 1001
    e = 100000
    dv = 0.000336499
    sqv = np.genfromtxt("sqv.csv", delimiter=",")
    co = np.ones(nv)
    co = 2 * dv * co

    max = 50  # could try a bit smaller like 20 or larger like 100
    it = 100  # number of iterations, as large as possible, millions...

    si = np.ones(nv)
    for _ in range(it):
        s = np.random.randint(2, size=nv)
        for _ in range(max):
            bv = np.dot(sqv, s) + np.sum(s) * 2 * dv

            for j in range(nv):
                if s[j] == 1 and bv[j] < 0:
                    s[j] = 0
                    es = np.dot(s, np.dot(sqv, s)) + 2 * dv * np.sum(s)
                    if es < e:
                        e = es
                        si = s.copy()

    print(e)
    print(si[:10], si[-10:])
    validate = np.loadtxt("validate.csv", delimiter=",")
    assert (validate == si).all()


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"time taken: {time.time() - t}")
