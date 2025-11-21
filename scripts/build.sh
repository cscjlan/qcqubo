#!/bin/bash

ml purge

ml PrgEnv-cray/8.5.0
ml craype-accel-amd-gfx90a
ml rocm/6.0.3

ml

set -exu
debug_flags="-xhip --std=c++20 -O0 -ggdb"
release_flags="-xhip --std=c++20 -O3 -DQUBO_DISABLE_ASSERTS"
flags=$release_flags

if ! command -v CC &> /dev/null
then
    CC=hipcc
else
    CC=CC
fi

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
src_dir="$script_dir/../src"
bin_dir="$script_dir/../bin"

$CC $flags -o $bin_dir/qubo -Isrc  $src_dir/main.cpp
#$CC $flags -o $bin_dir/tests -Isrc $src_dir/tests.cpp
