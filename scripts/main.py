import numpy as np
from numpy.ma import minimum
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


def generate_sparse_symmetric_real_matrix(n):
    mat = np.random.random((n, n))
    mat = mat + mat.T - 1.0

    prob = 1.0 - np.sqrt(1.0 - np.log10(n) ** 4 / n)
    mask = np.random.binomial(1, prob, (n, n))
    mask = ((mask + mask.T) > 0).astype(np.int64)

    return sparse.csr_array(mask * mat)


def generate_candidates(sq, num_vectors):
    randoms = np.random.randint(0, 2, size=(num_vectors, sq.shape[0]))
    return [Candidate(sq, x) for x in randoms]


class Greedy:
    def __init__(self, sq):
        self.diagonal = sq.diagonal()
        self.i = 0

    def keep_searching(self, candidate):
        sign = -2 * candidate.x + 1
        delta = sign * (2.0 * candidate.energies + sign * self.diagonal)
        self.i = np.argmin(delta)

        return delta[self.i] < 0.0

    def index(self):
        return self.i


class Candidate:
    def __init__(self, sq, x):
        self.x = x
        self.energies = sq @ x

    def update(self, i, sq):
        # sq is a sparse array
        begin = sq.indptr[i]
        end = sq.indptr[i + 1]

        row_indices = sq.indices[begin:end]
        row_data = sq.data[begin:end]

        # Update energies for rows
        delta = row_data * (-2 * self.x[i] + 1)
        self.energies[row_indices] += delta

        # Flip bit i
        self.x[i] = abs(self.x[i] - 1)

    def total_energy(self):
        return np.dot(self.x, self.energies)


def search(sq, candidates, local_search):
    num_iters = np.zeros(len(candidates))
    for ci, candidate in enumerate(candidates):
        while local_search.keep_searching(candidate):
            candidate.update(local_search.index(), sq)
            num_iters[ci] += 1

    return num_iters


def main():
    np.random.seed(24)
    print("Generating matrix")
    sq = generate_sparse_symmetric_real_matrix(10000)

    print("Generating candidates")
    candidates = generate_candidates(sq, 2000)

    print("Searching")
    search(sq, candidates, Greedy(sq))
    candidates.sort(key=lambda candidate: candidate.total_energy())
    minimum_candidate = candidates[0]

    # Print minimum energies for sanity check
    for c in candidates[:10]:
        e = c.total_energy()
        assert abs(e - c.x @ sq @ c.x) < 1e-9
        print(f"energy: {e}")

    # Write generated data to files for testing against C++-version
    with open("data/sparse_matrix.csv", "w") as f:
        sq.indptr.tofile(f, sep=",")
        f.write("\n")
        sq.indices.tofile(f, sep=",")
        f.write("\n")
        sq.data.tofile(f, sep=",")

    with open("data/candidates.csv", "w") as f:
        for c in candidates:
            c.x.astype(np.uint8).tofile(f, sep=",")
            f.write("\n")

    with open("data/minimum.csv", "w") as f:
        minimum_candidate.x.tofile(f, sep=",")
        f.write("\n")
        f.write(str(minimum_candidate.total_energy()))


if __name__ == "__main__":
    # sq = generate_sparse_symmetric_real_matrix(20000)
    # with open("data/large_matrix.csv", "w") as f:
    #    sq.indptr.tofile(f, sep=",")
    #    f.write("\n")
    #    sq.indices.tofile(f, sep=",")
    #    f.write("\n")
    #    sq.data.tofile(f, sep=",")
    main()
