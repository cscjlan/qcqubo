## Build on LUMI

1. Clone this repository to a directory: `git clone https://github.com/cscjlan/qcqubo.git`
2. `cd qcqubo`
3. Create a directory for binaries: `mkdir bin`
4. Run `scripts/build.sh` on e.g a login node. This builds an executable `qubo` in the `bin/` directory.

## Run on LUMI

First, use the `data/inputs.json` to specify the inputs to the program.
Specifically:
- `seed`: what random seed to use
- `num_to_search`: how many random binary vectors to generate and test
- `matrix_filename`: the `.csv` file containing the matrix, either in [csr format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) or in "dense" format,
    i.e. containing all the elements, including the zero elements
- `output_filename`: where to output the best binary vector

After you're satisfied with the inputs, use `sbatch -A project_YOUR_PROJECT_NUMBER scripts/lumi_run.sh`
to queue the program, where `YOUR_PROJECT_NUMBER` is a number of the project you want the billing of the used
resources to go to.
