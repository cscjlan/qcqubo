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
        s0 = np.random.randint(2, size=nv)
        s1 = s0.copy()
        for _ in range(max):
            sx = s1.copy()
            for j in range(nv):
                up = (s1[j] + 1) % 2
                se = sqv[j]
                bv = np.dot(se, sx) + np.dot(co, sx)
                if up == 1:
                    du = -bv
                else:
                    du = bv
                    if du < 0:
                        s1[j] = up
                        es = np.dot(s1, np.dot(sqv, s1)) + 2 * dv * np.dot(s1, s1)
                        if es < e:
                            e = es
                            si = s1.copy()

    print(e)
    print(si[:10], si[-10:])
    validate = np.loadtxt("validate.csv", delimiter=",")
    assert (validate == si).all()


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"time taken: {time.time() - t}")
