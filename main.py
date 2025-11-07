import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


def generate_sparse_symmetric_real_matrix(n):
    # generate a sparse symmetric real matrix
    print("generating")
    prob = np.log10(n) ** 4 / n
    mat = sparse.lil_array((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i, j] = 2.0 * (np.random.random() - 0.5)
                continue

            val = np.random.random()
            if val < prob:
                val /= 2.0 * (prob - 0.5)
                mat[i, j] = val
                mat[j, i] = val

    return sparse.csr_array(mat)


def integer_to_binary_vector(integer, num_bits):
    return np.array(
        [(integer >> (num_bits - position - 1)) & 1 for position in range(num_bits)]
    )


def test_integer_to_binary_vec(vec, integer, num_bits):
    converted = integer_to_binary_vector(integer, num_bits)
    for i, value in enumerate(vec):
        assert value == converted[i]


def evaluate(x, q):
    return x.T @ q @ x


def brute_force_minimum(q):
    vec_size = q.shape[0]
    num_values = 2**vec_size
    minimum_value = np.inf
    min_x = np.zeros(vec_size)
    for i in range(num_values):
        x = integer_to_binary_vector(i, vec_size)
        value = evaluate(x, q)
        if value < minimum_value:
            minimum_value = value
            min_x = x

    return (num_values, minimum_value, min_x)


def greedy_search(x, q):
    vec_size = q.shape[0]
    min_x = x.copy()
    minimum_value = evaluate(min_x, q)

    num_iters = 0
    while True:
        num_iters += 1
        min_i = -1
        for i in range(vec_size):
            min_x[i] = abs(min_x[i] - 1)
            value = evaluate(min_x, q)
            if value < minimum_value:
                minimum_value = value
                min_i = i
            min_x[i] = abs(min_x[i] - 1)

        # flip the minimum bit
        if min_i > 0:
            min_x[min_i] = abs(min_x[min_i] - 1)
            continue

        break

    return (minimum_value, min_x, num_iters)


def eig_minimum(q, values, vectors):
    vec_size = q.shape[0]
    i = 0

    minimum_value = np.inf
    min_x = np.zeros(vec_size)

    while i < values.shape[0] and values[i] < 0:
        eigvec = vectors[:, i]

        # Test the direction for which all values in the eigenvector are negative
        # value, x = greedy_search((eigvec < 0).astype(np.int64), q)
        x = (eigvec < 0).astype(np.int64)
        value = evaluate(x, q)

        if value < minimum_value:
            minimum_value = value
            min_x = x

        # Test the direction for which all values in the eigenvector are positive
        # value, x = greedy_search((eigvec > 0).astype(np.int64), q)
        x = (eigvec > 0).astype(np.int64)
        value = evaluate(x, q)

        if value < minimum_value:
            minimum_value = value
            min_x = x

        i += 1

    # Perform greedy search until we reach a minimum
    value, x, num_greedy_iters = greedy_search(min_x, q)
    if value < minimum_value:
        minimum_value = value
        min_x = x

    return (i, minimum_value, min_x, num_greedy_iters)


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


def sparse_eigen_greedy(sq, vectors):
    print("computing eigenvalues")
    candidates = np.hstack(
        ((vectors < 0).astype(np.int64), (vectors > 0).astype(np.int64))
    )
    minimum_value = np.inf
    min_x = np.zeros(sq.shape[0])

    half_diag = 0.5 * sq.diagonal()
    num_iters = 0
    for x in candidates.T:
        energies = (sq @ x) - 2.0 * half_diag * x

        while True:
            num_iters += 1
            delta = (energies + half_diag) * (-2 * x + 1)
            i = np.argmin(delta)

            if delta[i] >= 0:
                break

            # Find the indices and data for matrix row i
            begin = sq.indptr[i]
            end = sq.indptr[i + 1]
            row_indices = sq.indices[begin:end]
            row_data = sq.data[begin:end]

            # Remove the diagonal
            column_indices = row_indices[row_indices != i]
            column_data = row_data[row_indices != i]

            # Update the row energies
            # This works due to symmetry of sq
            energies[column_indices] += column_data

            x[i] = abs(x[i] - 1)

        value = x @ sq @ x
        if value < minimum_value:
            minimum_value = value
            min_x = x

    return (num_iters, minimum_value, min_x)


def with_small(q, values, vectors):
    brute_num_values, brute_minimum_value, brute_min_x = brute_force_minimum(q)
    eig_num_values, eig_minimum_value, eig_min_x, num_greedy_iters = eig_minimum(
        q, values, vectors
    )

    print(f"brute minimum: {brute_minimum_value}, eig minimum: {eig_minimum_value}")
    print(
        f"Brute values searched: {brute_num_values}, eig values searched: {eig_num_values}, greedy iters: {num_greedy_iters}"
    )
    print("Bit vectors (brute first, eig second)")
    print(brute_min_x)
    print(eig_min_x)


def main():
    # TODO for some reason this always returns vector full of ones
    np.random.seed(22234)
    # sqv = np.genfromtxt("sqv.csv", delimiter=",")
    n = 200
    sq = generate_sparse_symmetric_real_matrix(n)
    values, vectors = linalg.eigsh(sq, np.min([2, sq.shape[0] - 1]), which="SA")

    if n <= 22:
        q = sq.toarray()
        with_small(q, values, vectors)

    sge_num_iters, sge_minimum_value, sge_min_x = sparse_eigen_greedy(sq, vectors)
    lower_limit = compute_lower_limit(sge_min_x, values[0])
    print(f"sge_num_iters: {sge_num_iters}")
    print(f"sge_minimum_value: {sge_minimum_value}, lower limit: {lower_limit}")
    print(f"sge_min_x: {sge_min_x}")


if __name__ == "__main__":
    test_integer_to_binary_vec([0, 0, 0, 1], 1, 4)
    test_integer_to_binary_vec([0, 0, 1, 0], 2, 4)
    test_integer_to_binary_vec([0, 0, 1, 1], 3, 4)
    test_integer_to_binary_vec([1, 0, 1, 1], 11, 4)

    main()
