import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


def generate_sparse_symmetric_real_matrix(n):
    print(f"Generating a sparse symmetric real matrix of size {n}x{n}")
    mat = np.random.random((n, n))
    mat = mat + mat.T - 1.0

    prob = 1.0 - np.sqrt(1.0 - np.log10(n) ** 4 / n)
    mask = np.random.binomial(1, prob, (n, n))
    mask = ((mask + mask.T) > 0).astype(np.int64)

    return sparse.csr_array(mask * mat)


def compute_lower_limit(x, eigenvalue):
    # Compute the lower limit of the minimum, i.e. the value of the function
    # at the direction of the eigenvector corresponding to the minimum eigenvalue
    # given a vector of the same length as the found min_x.
    # In other words, evaluate D_{ii} * (p_i . v)^2, where
    # D_{ii} is the smallest eigenvalue
    # p_i is the corresponding eigenvector
    # v || p_i, i.e. v and p_i point in the same direction
    # length(v) == length(min_x)
    # and . is the dot product
    return np.sum(x) * eigenvalue


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


class Travel:
    def __init__(self, target):
        self.target = target
        self.i = 0

    def keep_searching(self, candidate):
        values = np.nonzero(candidate.x - self.target)[0]
        any_different = len(values) > 0
        if any_different:
            self.i = values[0]

        return any_different

    def index(self):
        return self.i


class Candidate:
    def __init__(self, sq, x, name):
        self.name = name
        self.x = x
        self.minimum_x = x.copy()
        self.energies = sq @ x
        self.total_energy = np.dot(self.x, self.energies)
        self.minimum_energy = self.total_energy

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

        self.total_energy = np.dot(self.energies, self.x)

        if self.total_energy < self.minimum_energy:
            self.minimum_energy = self.total_energy
            self.minimum_x = self.x.copy()


def search(sq, candidates, local_search):
    num_iters = np.zeros(len(candidates))
    for ci, candidate in enumerate(candidates):
        while local_search.keep_searching(candidate):
            candidate.update(local_search.index(), sq)
            num_iters[ci] += 1

    return num_iters


def generate_initial_candidates(sq):
    num_eigenvalues = 100
    num_eigenvalues = np.min((int(sq.shape[0] / 2), num_eigenvalues))
    _, vectors = linalg.eigsh(sq, num_eigenvalues, which="SA")
    sum_vec = np.sum(vectors, axis=1).reshape((vectors.shape[0], 1))

    randoms = np.random.randint(0, 2, size=(vectors.shape[0], num_eigenvalues))

    candidate_xs = np.hstack(
        (
            (vectors < 0).astype(np.int64),
            (vectors > 0).astype(np.int64),
            (sum_vec < 0).astype(np.int64),
            (sum_vec > 0).astype(np.int64),
            randoms,
        )
    )

    def make_name(i, num_eigenvalues):
        if i <= num_eigenvalues:
            return "-" + str(i)
        elif i <= 2 * num_eigenvalues:
            return "+" + str(i - num_eigenvalues)
        elif i <= 2 * num_eigenvalues + 1:
            return "-s"
        elif i <= 2 * num_eigenvalues + 2:
            return "+s"

        return "r" + str(i - 2 * (num_eigenvalues + 1))

    return [
        Candidate(sq, x, make_name(i + 1, num_eigenvalues))
        for i, x in enumerate(candidate_xs.T)
    ]


def main():
    np.random.seed(24)
    # sq = sparse.csr_array(np.genfromtxt("sqv.csv", delimiter=","))
    sq = generate_sparse_symmetric_real_matrix(1000)

    print("Generating candidates")
    candidates = generate_initial_candidates(sq)

    print("Searching around eigenvectors")
    # num_iters = search(sq, candidates, Greedy(sq))
    num_iters = search(sq, candidates, Travel(np.ones(sq.shape[0])))
    candidates.sort(key=lambda candidate: candidate.total_energy)
    minimum_candidate = candidates[0]

    print("Computing lower limit for minimum value")
    values, _ = linalg.eigsh(sq, 1, which="SA")
    lower_limit = compute_lower_limit(minimum_candidate.minimum_x, values[0])

    print(f"number of iterations: {np.sum(num_iters)}")
    print(f"lower limit: {lower_limit}")
    for c in candidates[:10]:
        print(
            f"{c.name}: {c.total_energy}, {np.sqrt(np.dot(candidates[0].minimum_x - c.minimum_x, candidates[0].minimum_x - c.minimum_x))}"
        )
    print(
        f"sge_min_x: {minimum_candidate.minimum_x[:10]}...{minimum_candidate.minimum_x[-10:]}"
    )


if __name__ == "__main__":
    np.random.seed(24)
    # sq = sparse.csr_array(np.genfromtxt("sqv.csv", delimiter=","))
    for i in range(1):
        sq = generate_sparse_symmetric_real_matrix(200)
        with open("generated_" + str(i) + ".csv", "w") as f:
            sq.indptr.tofile(f, sep=",")
            f.write("\n")
            sq.indices.tofile(f, sep=",")
            f.write("\n")
            sq.data.tofile(f, sep=",")
    # main()
